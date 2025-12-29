#!/usr/bin/env python3
"""
오프라인 평가 스크립트

저장된 CSV 파일들을 로드하여 메트릭을 계산합니다.
기존 on_evaluation_epoch_end -> per_device_evaluate 흐름과 별도로 작동합니다.

사용법:
    python offline_evaluate.py --csv_dir results/20251217 --script_name galactica_test
    python offline_evaluate.py --csv_path results/20251217/120000_galactica_test_classification_merged.csv
"""

import argparse
import os
import glob
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta

# 메트릭 계산용 라이브러리
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import selfies as sf

# Task 정의 import
from model.result_saver import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    MOLECULE_GENERATION_TASKS,
    CAPTIONING_TASKS,
    get_task_type,
)


def evaluate_classification(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Classification CSV에서 메트릭 계산

    일반적인 관행:
    - AUC-ROC: prob >= 0인 유효한 샘플만 사용 (파싱 실패 제외)
    - Accuracy/F1/Precision/Recall: 파싱 실패는 틀린 것으로 처리 (pred=0)
    - failure_rate: 파싱 실패 비율 별도 리포트

    필요 컬럼: idx, task, label, pred, prob, correct
    """
    results = {}

    for task in df['task'].unique():
        subset = df[df['task'] == task]

        # label 파싱
        labels = np.array([1 if 'True' in str(l) else 0 for l in subset['label']])

        # prob 컬럼 사용
        probs = pd.to_numeric(subset['prob'], errors='coerce').values

        # 파싱 실패 (prob < 0) 마스크
        valid_mask = probs >= 0
        failure_rate = 1.0 - valid_mask.mean()

        # Accuracy/F1 등: CSV의 pred 컬럼 직접 사용 (온라인과 동일)
        preds = subset['pred'].values.astype(int)

        # sklearn 메트릭 계산 (전체 샘플 - 파싱 실패는 틀린 것으로 처리)
        try:
            acc = accuracy_score(y_true=labels, y_pred=preds)
            f1 = f1_score(y_true=labels, y_pred=preds)
            prec = precision_score(y_true=labels, y_pred=preds)
            rec = recall_score(y_true=labels, y_pred=preds)
        except Exception as e:
            print(f"[{task}] sklearn metric error: {e}")
            acc = f1 = prec = rec = float('nan')

        # AUC-ROC: 유효한 prob만 사용 (파싱 실패 제외)
        # 온라인 평가처럼 조건 체크 없이 바로 계산 (에러 시에만 nan)
        valid_labels = labels[valid_mask]
        valid_probs = probs[valid_mask]
        try:
            roc_auc = roc_auc_score(y_true=valid_labels, y_score=valid_probs)
        except Exception as e:
            print(f"[{task}] ROC-AUC error: {e}")
            roc_auc = float('nan')

        # AUC-ROC (all): 전체 포함 (온라인 평가와 동일)
        # prob=-1인 경우도 그대로 사용, 조건 체크 없음
        try:
            roc_auc_all = roc_auc_score(y_true=labels, y_score=probs)
        except Exception as e:
            print(f"[{task}] ROC-AUC (all) error: {e}")
            roc_auc_all = float('nan')

        results[task] = {
            'accuracy': acc,
            'f1': f1,
            'precision': prec,
            'recall': rec,
            'roc_auc': roc_auc,
            'roc_auc_all': roc_auc_all,
            'num_samples': len(subset),
            'failure_rate': failure_rate,
        }

    return results


def evaluate_regression(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Regression CSV에서 메트릭 계산

    필요 컬럼: idx, task, label, pred, error, parse_failed
    """
    results = {}

    for task in df['task'].unique():
        subset = df[df['task'] == task]

        # parse_failed=1인 경우 제외 (온라인 평가와 동일)
        # parse_failed 컬럼이 없는 경우 (기존 CSV) → pred가 숫자인지로 판단
        if 'parse_failed' in subset.columns:
            valid_subset = subset[subset['parse_failed'] != 1]
        else:
            # 기존 CSV 호환: pred가 숫자로 변환 가능한 경우만 유효
            valid_subset = subset[pd.to_numeric(subset['pred'], errors='coerce').notna()]

        failure_rate = 1 - len(valid_subset) / len(subset) if len(subset) > 0 else 0

        if len(valid_subset) > 0:
            # error 컬럼이 NaN이 아닌 것만 사용
            valid_errors = valid_subset[valid_subset['error'].notna()]
            if len(valid_errors) > 0:
                errors = valid_errors['error'].values
                mae = np.mean(np.abs(errors))
                mse = np.mean(errors ** 2)
                rmse = np.sqrt(mse)
            else:
                mae = mse = rmse = float('nan')
        else:
            mae = mse = rmse = float('nan')

        results[task] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'failure_rate': failure_rate,
            'num_samples': len(subset),
        }

    return results


def parse_smiles_from_label(label: str) -> str:
    """label에서 SMILES 추출 (태그 제거, SELFIES는 canonical SMILES로 변환)"""
    import re
    try:
        import selfies as sf
        from rdkit import Chem
    except ImportError:
        sf = None
        Chem = None

    if not label:
        return ""

    label = str(label).strip()

    # <SELFIES>...</SELFIES> 형식 -> canonical SMILES로 변환
    match = re.search(r'<SELFIES>\s*(.*?)\s*</SELFIES>', label, re.DOTALL)
    if match and sf:
        selfies_str = match.group(1).strip()
        try:
            smiles = sf.decoder(selfies_str)
            if Chem:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    return Chem.MolToSmiles(mol)  # canonical SMILES
            return smiles
        except:
            return selfies_str

    # <SMILES>...</SMILES> 형식
    match = re.search(r'<SMILES>\s*(.*?)\s*</SMILES>', label, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 태그 없이 바로 SMILES인 경우
    # EOS 토큰 제거
    for eos_token in ['</s>', '<|end_of_text|>', '<|endoftext|>', '<|eot_id|>']:
        if eos_token in label:
            label = label.split(eos_token)[0]

    return label.strip()


def calculate_fcd(gt_smiles_list: List[str], pred_smiles_list: List[str]) -> float:
    """
    FCD (Fréchet ChEMBL Distance) 계산

    Args:
        gt_smiles_list: Ground truth SMILES 리스트
        pred_smiles_list: Predicted SMILES 리스트

    Returns:
        FCD score (낮을수록 좋음, 0 = 동일한 분포)
    """
    try:
        from fcd import get_fcd, load_ref_model
    except ImportError:
        print("[WARNING] fcd package not installed. Run: pip install fcd")
        return float('nan')

    if len(gt_smiles_list) < 2 or len(pred_smiles_list) < 2:
        print("[WARNING] Not enough valid SMILES for FCD calculation (need at least 2)")
        return float('nan')

    try:
        # CPU에서만 실행 (오프라인 평가는 GPU 사용 안함)
        model = load_ref_model(device='cpu')
        fcd_score = get_fcd(gt_smiles_list, pred_smiles_list, model, device='cpu')
        return fcd_score
    except Exception as e:
        print(f"[WARNING] FCD calculation failed: {e}")
        return float('nan')


def calculate_bleu_selfies(gt_smiles_list: List[str], pred_smiles_list: List[str]) -> float:
    """
    BLEU-SELFIES 계산 (SMILES → SELFIES 변환 후 토큰 단위 BLEU)

    Args:
        gt_smiles_list: Ground truth SMILES 리스트 (canonical)
        pred_smiles_list: Predicted SMILES 리스트 (canonical)

    Returns:
        BLEU-SELFIES score (0-100, 높을수록 좋음)
    """
    if not gt_smiles_list or not pred_smiles_list:
        return 0.0

    ref_selfies_list = []
    pred_selfies_list = []

    for gt_smiles, pred_smiles in zip(gt_smiles_list, pred_smiles_list):
        try:
            # SMILES → SELFIES 변환
            gt_selfies = sf.encoder(gt_smiles)
            pred_selfies = sf.encoder(pred_smiles)

            if gt_selfies and pred_selfies:
                # SELFIES 토큰화 (각 [...] 단위로 분리)
                gt_tokens = list(sf.split_selfies(gt_selfies))
                pred_tokens = list(sf.split_selfies(pred_selfies))

                ref_selfies_list.append([gt_tokens])  # corpus_bleu는 ref가 list of list 형태
                pred_selfies_list.append(pred_tokens)
        except Exception:
            continue

    if not pred_selfies_list:
        return 0.0

    try:
        bleu_selfies = corpus_bleu(
            ref_selfies_list, pred_selfies_list,
            weights=(0.25, 0.25, 0.25, 0.25)
        )
        return bleu_selfies * 100  # 0-100 스케일
    except Exception as e:
        print(f"[WARNING] BLEU-SELFIES calculation failed: {e}")
        return 0.0


def evaluate_molecule_generation(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Molecule Generation CSV에서 메트릭 계산

    온라인 평가와 동일:
    - validity_ratio: 전체 샘플 중 유효한 분자 비율
    - exact_match, fingerprint sims, levenshtein: 유효한 분자만 평균
    - text_exact_match: 텍스트 관점에서 정확히 일치하는 비율 (canonical화 없이)
    - fcd: Fréchet ChEMBL Distance (유효한 분자만 사용)

    필요 컬럼: idx, task, label, pred, validity, exact_match, MACCS_FTS, RDK_FTS, morgan_FTS, levenshtein
    """
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')  # RDKit 경고 메시지 비활성화
    except ImportError:
        Chem = None

    results = {}

    # 전체 task에 대한 FCD 계산을 위한 SMILES 수집
    all_gt_smiles = []
    all_pred_smiles = []

    for task in df['task'].unique():
        subset = df[df['task'] == task]

        # validity_ratio: 전체 샘플 중 유효한 분자 비율
        validity_ratio = subset['validity'].mean()

        # 유효한 분자만 추출 (온라인 평가와 동일)
        valid_subset = subset[subset['validity'] == 1]

        if len(valid_subset) > 0:
            # exact_match, fingerprint sims, levenshtein: 유효한 분자만 평균
            exact_match_ratio = valid_subset['exact_match'].mean()
            maccs_fts = valid_subset['MACCS_FTS'].mean()
            rdk_fts = valid_subset['RDK_FTS'].mean()
            morgan_fts = valid_subset['morgan_FTS'].mean()
            levenshtein = valid_subset['levenshtein'].mean()
        else:
            exact_match_ratio = float('nan')
            maccs_fts = rdk_fts = morgan_fts = levenshtein = float('nan')

        # text_exact_match: 텍스트 관점에서 정확히 일치하는 비율 (전체 샘플 대상)
        text_matches = 0

        # FCD 계산을 위한 SMILES 수집
        task_gt_smiles = []
        task_pred_smiles = []

        for _, row in subset.iterrows():
            label_smiles = parse_smiles_from_label(row['label'])
            pred_smiles = str(row['pred']).strip() if pd.notna(row['pred']) else ""

            if label_smiles == pred_smiles:
                text_matches += 1

            # FCD용 SMILES 수집 (유효한 분자만)
            if Chem and label_smiles and pred_smiles:
                try:
                    gt_mol = Chem.MolFromSmiles(label_smiles)
                    pred_mol = Chem.MolFromSmiles(pred_smiles)
                    if gt_mol and pred_mol:
                        canonical_gt = Chem.MolToSmiles(gt_mol)
                        canonical_pred = Chem.MolToSmiles(pred_mol)
                        task_gt_smiles.append(canonical_gt)
                        task_pred_smiles.append(canonical_pred)
                        all_gt_smiles.append(canonical_gt)
                        all_pred_smiles.append(canonical_pred)
                except:
                    pass

        text_exact_match_ratio = text_matches / len(subset) if len(subset) > 0 else 0

        # Task별 FCD 계산
        print(f"[{task}] Calculating FCD with {len(task_gt_smiles)} valid SMILES pairs...")
        task_fcd = calculate_fcd(task_gt_smiles, task_pred_smiles)

        # Task별 BLEU-SELFIES 계산
        task_bleu_selfies = calculate_bleu_selfies(task_gt_smiles, task_pred_smiles)

        results[task] = {
            'validity_ratio': validity_ratio,
            'exact_match_ratio': exact_match_ratio,
            'text_exact_match_ratio': text_exact_match_ratio,
            'MACCS_FTS': maccs_fts,
            'RDK_FTS': rdk_fts,
            'morgan_FTS': morgan_fts,
            'levenshtein_score': levenshtein,
            'bleu_selfies': task_bleu_selfies,
            'fcd': task_fcd,
            'num_samples': len(subset),
            'num_valid_for_fcd': len(task_gt_smiles),
        }

    # 전체 molecule generation task에 대한 통합 FCD 및 BLEU-SELFIES
    if len(all_gt_smiles) > 0:
        print(f"\n[ALL MOLECULE GENERATION] Calculating overall FCD with {len(all_gt_smiles)} valid SMILES pairs...")
        overall_fcd = calculate_fcd(all_gt_smiles, all_pred_smiles)
        overall_bleu_selfies = calculate_bleu_selfies(all_gt_smiles, all_pred_smiles)
        results['_overall_fcd'] = {
            'fcd': overall_fcd,
            'bleu_selfies': overall_bleu_selfies,
            'num_valid_pairs': len(all_gt_smiles),
        }

    return results


def parse_description(text: str) -> str:
    """<DESCRIPTION>...</DESCRIPTION> 태그에서 내용 추출"""
    import re
    if not text:
        return ""

    # <DESCRIPTION>...</DESCRIPTION> 형식
    match = re.search(r'<DESCRIPTION>\s*(.*?)\s*</DESCRIPTION>', str(text), re.DOTALL)
    if match:
        return match.group(1).strip()

    # Left-side only
    match = re.search(r'<DESCRIPTION>\s*(.*)', str(text), re.DOTALL)
    if match:
        return match.group(1).strip()

    return str(text).strip()


def parse_caption_pred(text: str) -> str:
    """Captioning prediction 후처리 - CSV의 pred는 이미 파싱된 텍스트이므로 그대로 반환"""
    if not text:
        return ""
    return str(text).strip()


def evaluate_captioning(df: pd.DataFrame, tokenizer=None) -> Dict[str, Any]:
    """
    Captioning CSV에서 메트릭 계산 (corpus 단위)

    필요 컬럼: idx, task, label, pred

    Note: BLEU, ROUGE, METEOR는 corpus 단위로 재계산해야 함
    """
    results = {}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

    for task in df['task'].unique():
        subset = df[df['task'] == task]

        references = []
        hypotheses = []
        ref_sentences = []
        hyp_sentences = []

        for _, row in subset.iterrows():
            # label에서 <DESCRIPTION> 태그 내용 추출
            ref = parse_description(row['label'])
            # pred 후처리
            hyp = parse_caption_pred(row['pred'])

            if not ref or not hyp:
                continue

            # character-level 토크나이징 (온라인 평가와 동일)
            ref_tokens = list(ref)
            hyp_tokens = list(hyp)

            references.append([ref_tokens])
            hypotheses.append(hyp_tokens)
            ref_sentences.append(ref)
            hyp_sentences.append(hyp)

        # BLEU
        if hypotheses:
            bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5)) * 100
            bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)) * 100
        else:
            bleu2 = bleu4 = 0

        # METEOR
        meteor_scores = []
        for ref, hyp in tqdm(zip(references, hypotheses), desc=f"METEOR for {task}", total=len(references)):
            mscore = meteor_score(ref, hyp)
            meteor_scores.append(mscore)
        _meteor_score = np.mean(meteor_scores) * 100 if meteor_scores else 0

        # ROUGE
        rouge_scores = []
        for ref_sen, hyp_sen in tqdm(zip(ref_sentences, hyp_sentences), desc=f"ROUGE for {task}", total=len(ref_sentences)):
            lscore = scorer.score(hyp_sen, ref_sen)
            rouge_scores.append(lscore)

        if rouge_scores:
            rouge1 = np.mean([rs["rouge1"].fmeasure for rs in rouge_scores]) * 100
            rouge2 = np.mean([rs["rouge2"].fmeasure for rs in rouge_scores]) * 100
            rougeL = np.mean([rs["rougeL"].fmeasure for rs in rouge_scores]) * 100
        else:
            rouge1 = rouge2 = rougeL = 0

        results[task] = {
            'bleu2': bleu2,
            'bleu4': bleu4,
            'meteor': _meteor_score,
            'rouge1': rouge1,
            'rouge2': rouge2,
            'rougeL': rougeL,
            'num_samples': len(subset),
        }

    return results


def load_csv_files(csv_dir: str, script_name: str) -> Dict[str, pd.DataFrame]:
    """
    디렉토리에서 CSV 파일 로드

    script_name 형식:
    - "galactica_test": 해당 이름 포함 모든 파일 (*_galactica_test_rank*.csv)
    - "152204_galactica_test": timestamp 포함 정확한 매칭 (152204_galactica_test_rank*.csv)

    1. 통합 CSV (새 구조): {script_name}_rank*.csv (metakey 컬럼 있음)
    2. task type별 CSV (기존 구조): {script_name}_{task_type}_rank*.csv
    """
    dfs = {}

    # script_name이 숫자로 시작하면 timestamp 포함 - 정확한 prefix 매칭
    # 그렇지 않으면 *_{script_name} 패턴 사용
    if script_name[0].isdigit():
        # timestamp 포함: 152204_galactica_test -> 152204_galactica_test_rank*.csv
        unified_pattern = os.path.join(csv_dir, f"{script_name}_rank*.csv")
    else:
        # timestamp 미포함: galactica_test -> *_galactica_test_rank*.csv
        unified_pattern = os.path.join(csv_dir, f"*_{script_name}_rank*.csv")

    # 1. 먼저 통합 CSV 파일 찾기 (새 구조)
    unified_files = [f for f in glob.glob(unified_pattern)
                     if not any(tt in os.path.basename(f) for tt in ['_classification_', '_regression_', '_molecule_generation_', '_captioning_'])]

    if unified_files:
        print(f"[Unified] Found {len(unified_files)} unified CSV files")
        df_list = [pd.read_csv(f) for f in sorted(unified_files)]
        merged = pd.concat(df_list, ignore_index=True)
        # 중복 제거 비활성화 - 온라인 평가와 동일하게 DDP 중복 포함
        # merged = merged.drop_duplicates(subset=['idx', 'task'], keep='first')
        merged = merged.sort_values('idx').reset_index(drop=True)
        print(f"[Unified] Total {len(merged)} samples loaded")

        # task별로 분리
        for task in merged['task'].unique():
            task_df = merged[merged['task'] == task].copy()
            task_type = get_task_type(task)
            if task_type not in dfs:
                dfs[task_type] = task_df
            else:
                dfs[task_type] = pd.concat([dfs[task_type], task_df], ignore_index=True)

        for task_type, df in dfs.items():
            print(f"  [{task_type}] {len(df)} samples, tasks: {df['task'].unique().tolist()}")

        return dfs

    # 2. 기존 task type별 CSV 구조
    task_types = ['classification', 'regression', 'molecule_generation', 'captioning']

    for task_type in task_types:
        # timestamp 포함 여부에 따라 패턴 결정
        if script_name[0].isdigit():
            merged_pattern = os.path.join(csv_dir, f"{script_name}_{task_type}_merged.csv")
            rank_pattern = os.path.join(csv_dir, f"{script_name}_{task_type}_rank*.csv")
        else:
            merged_pattern = os.path.join(csv_dir, f"*_{script_name}_{task_type}_merged.csv")
            rank_pattern = os.path.join(csv_dir, f"*_{script_name}_{task_type}_rank*.csv")

        # merged 파일 찾기
        merged_files = glob.glob(merged_pattern)

        if merged_files:
            dfs[task_type] = pd.read_csv(merged_files[0])
            print(f"[{task_type}] Loaded merged file: {merged_files[0]}")
        else:
            # rank별 파일 찾기
            rank_files = sorted(glob.glob(rank_pattern))

            if rank_files:
                df_list = [pd.read_csv(f) for f in rank_files]
                merged = pd.concat(df_list, ignore_index=True)
                # 중복 제거 비활성화 - 온라인 평가와 동일하게 DDP 중복 포함
                # merged = merged.drop_duplicates(subset=['idx'], keep='first')
                merged = merged.sort_values('idx').reset_index(drop=True)
                dfs[task_type] = merged
                print(f"[{task_type}] Merged {len(rank_files)} rank files: {len(merged)} samples")
            else:
                print(f"[{task_type}] No files found")

    return dfs


def evaluate_all(dfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    모든 task type에 대해 평가 수행
    """
    all_results = {}

    if 'classification' in dfs and len(dfs['classification']) > 0:
        print("\n=== Evaluating Classification ===")
        all_results['classification'] = evaluate_classification(dfs['classification'])

    if 'regression' in dfs and len(dfs['regression']) > 0:
        print("\n=== Evaluating Regression ===")
        all_results['regression'] = evaluate_regression(dfs['regression'])

    if 'molecule_generation' in dfs and len(dfs['molecule_generation']) > 0:
        print("\n=== Evaluating Molecule Generation ===")
        all_results['molecule_generation'] = evaluate_molecule_generation(dfs['molecule_generation'])

    if 'captioning' in dfs and len(dfs['captioning']) > 0:
        print("\n=== Evaluating Captioning ===")
        all_results['captioning'] = evaluate_captioning(dfs['captioning'])

    return all_results


def format_results(results: Dict[str, Any]) -> str:
    """
    결과를 보기 좋게 포맷팅
    """
    lines = []

    for task_type, task_results in results.items():
        lines.append(f"\n{'='*60}")
        lines.append(f"  {task_type.upper()}")
        lines.append(f"{'='*60}")

        for task, metrics in task_results.items():
            lines.append(f"\n  [{task}]")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"    {metric}: {value:.4f}")
                else:
                    lines.append(f"    {metric}: {value}")

    return '\n'.join(lines)


def save_results(results: Dict[str, Any], output_path: str):
    """
    결과를 JSON으로 저장
    """
    # benchmark_performance.json과 유사한 형식으로 변환
    flat_results = {}

    for task_type, task_results in results.items():
        for task, metrics in task_results.items():
            for metric, value in metrics.items():
                if metric == 'num_samples':
                    continue
                key = f"test/{task}/{metric}"
                flat_results[key] = value

    with open(output_path, 'w') as f:
        json.dump(flat_results, f, indent=4)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='오프라인 평가 스크립트')
    parser.add_argument('--csv_dir', type=str, help='CSV 파일들이 있는 디렉토리')
    parser.add_argument('--script_name', type=str, help='스크립트 이름 (파일명 패턴)')
    parser.add_argument('--csv_path', type=str, help='단일 CSV 파일 경로')
    parser.add_argument('--output', type=str, default=None, help='결과 저장 경로 (JSON)')
    args = parser.parse_args()

    if args.csv_path:
        # 단일 파일 평가
        df = pd.read_csv(args.csv_path)

        # 파일명에서 task type 추출
        filename = os.path.basename(args.csv_path)
        if 'classification' in filename:
            task_type = 'classification'
            results = {'classification': evaluate_classification(df)}
        elif 'regression' in filename:
            task_type = 'regression'
            results = {'regression': evaluate_regression(df)}
        elif 'molecule_generation' in filename:
            task_type = 'molecule_generation'
            results = {'molecule_generation': evaluate_molecule_generation(df)}
        elif 'captioning' in filename:
            task_type = 'captioning'
            results = {'captioning': evaluate_captioning(df)}
        else:
            print("Unknown task type in filename")
            return

        print(f"Loaded {len(df)} samples from {args.csv_path}")

    elif args.csv_dir and args.script_name:
        # 디렉토리에서 모든 파일 로드
        dfs = load_csv_files(args.csv_dir, args.script_name)
        results = evaluate_all(dfs)
    else:
        parser.print_help()
        return

    # 결과 출력
    print(format_results(results))

    # 결과 저장
    if args.output:
        save_results(results, args.output)
    elif args.csv_dir:
        # 기본 저장 경로
        kst = timezone(timedelta(hours=9))
        now = datetime.now(kst)
        output_path = os.path.join(
            args.csv_dir,
            f"{now.strftime('%H%M%S')}_{args.script_name}_offline_evaluation.json"
        )
        save_results(results, output_path)


if __name__ == '__main__':
    main()
