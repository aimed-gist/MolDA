"""
ResultSaver: 샘플별 평가 결과를 통합 CSV로 저장하는 클래스

저장 경로: /workspace/Mol_DA_repo/Mol-LLM-Benchmark/results/{날짜}/{시간}_{스크립트명}_rank{rank}.csv
시간대: KST (UTC+9)

통합 컬럼 구조:
- 공통: idx, task, label, pred, metakey
- Classification: prob, correct
- Regression: error
- Molecule Generation: validity, exact_match, MACCS_FTS, RDK_FTS, morgan_FTS, levenshtein
- Captioning: (추가 metric 없음)
"""

import os
import csv
import yaml
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from argparse import Namespace


# 태스크 유형 정의
CLASSIFICATION_TASKS = [
    "smol-property_prediction-bbbp",
    "smol-property_prediction-clintox",
    "smol-property_prediction-hiv",
    "smol-property_prediction-sider",
    "bace",
    "tox21",
    "toxcast",
]

REGRESSION_TASKS = [
    "smol-property_prediction-esol",
    "smol-property_prediction-lipo",
    "qm9_homo",
    "qm9_lumo",
    "qm9_homo_lumo_gap",
    "qm9_dipole_moment",
    "qm9_isotropic_polarizability",
    "qm9_electronic_spatial_extent",
    "qm9_zero_point_vibrational_energy",
    "qm9_heat_capacity_298K",
    "qm9_internal_energy_298K",
    "qm9_enthalpy_298K",
    "qm9_free_energy_298K",
    "alchemy_homo",
    "alchemy_lumo",
    "alchemy_homo_lumo_gap",
    "aqsol-logS",
    "pcqm_homo_lumo_gap",
]

MOLECULE_GENERATION_TASKS = [
    "forward_reaction_prediction",
    "smol-forward_synthesis",
    "retrosynthesis",
    "smol-retrosynthesis",
    "reagent_prediction",
    "presto-forward_reaction_prediction",
    "presto-retrosynthesis",
    "presto-reagent_prediction",
    "orderly-forward_reaction_prediction",
    "orderly-retrosynthesis",
    "orderly-reagent_prediction",
    "chebi-20-text2mol",
    "smol-molecule_generation",
]

CAPTIONING_TASKS = [
    "chebi-20-mol2text",
    "smol-molecule_captioning",
]


# 통합 CSV 컬럼 정의
UNIFIED_COLUMNS = [
    "idx", "task", "label", "pred", "metakey",
    # Classification
    "prob", "correct",
    # Regression
    "error",
    # Molecule Generation
    "validity", "exact_match", "MACCS_FTS", "RDK_FTS", "morgan_FTS", "levenshtein",
]

# Task type별 metakey
CLASSIFICATION_METAKEYS = "prob,correct"
REGRESSION_METAKEYS = "error,parse_failed"
MOLECULE_GENERATION_METAKEYS = "validity,exact_match,MACCS_FTS,RDK_FTS,morgan_FTS,levenshtein"
CAPTIONING_METAKEYS = ""


def get_task_type(task: str) -> str:
    """태스크 이름으로 태스크 유형 반환"""
    if task in CLASSIFICATION_TASKS:
        return "classification"
    elif task in REGRESSION_TASKS:
        return "regression"
    elif task in MOLECULE_GENERATION_TASKS:
        return "molecule_generation"
    elif task in CAPTIONING_TASKS:
        return "captioning"
    else:
        return "unknown"


class ResultSaver:
    """샘플별 평가 결과를 통합 CSV로 실시간 저장하는 클래스"""

    def __init__(
        self,
        script_name: str,
        base_dir: str = "/workspace/Mol_DA_repo/Mol-LLM-Benchmark/results",
        rank: int = 0,
    ):
        """
        Args:
            script_name: 실행 스크립트 파일명 (e.g., "llasmol_test.sh")
            base_dir: 결과 저장 기본 경로
            rank: 분산 학습 시 rank (파일명에 포함)
        """
        self.script_name = self._clean_script_name(script_name)
        self.base_dir = base_dir
        self.rank = rank
        self.sample_count = 0

        # 저장 경로 초기화
        self._date_str: str = ""
        self._time_str: str = ""
        self._dir_path: str = ""
        self._save_path: str = ""
        self._header_written: bool = False
        self._init_save_path()

    def _clean_script_name(self, script_name: str) -> str:
        """스크립트 이름에서 확장자 제거"""
        script_name = os.path.basename(script_name)
        if script_name.endswith(".sh"):
            script_name = script_name[:-3]
        elif script_name.endswith(".py"):
            script_name = script_name[:-3]
        return script_name

    def _init_save_path(self):
        """KST 기준 날짜/시간으로 저장 경로 생성"""
        kst = timezone(timedelta(hours=9))
        now = datetime.now(kst)

        self._date_str = now.strftime("%Y%m%d")
        self._time_str = now.strftime("%H%M%S")

        self._dir_path = os.path.join(self.base_dir, self._date_str)
        os.makedirs(self._dir_path, exist_ok=True)

        # 통합 CSV 파일 경로
        filename = f"{self._time_str}_{self.script_name}_rank{self.rank}.csv"
        self._save_path = os.path.join(self._dir_path, filename)

        # Config 파일 경로
        config_filename = f"{self._time_str}_{self.script_name}_config.yaml"
        self._config_path = os.path.join(self._dir_path, config_filename)

    def save_config(self, args):
        """args를 YAML 파일로 저장 (rank 0에서만 한 번 호출)

        Args:
            args: argparse.Namespace 또는 OmegaConf DictConfig
        """
        if self.rank != 0:
            return  # rank 0에서만 저장

        # args를 dict로 변환
        if hasattr(args, '__dict__'):
            # argparse.Namespace
            config_dict = vars(args).copy()
        elif hasattr(args, 'to_container'):
            # OmegaConf DictConfig
            from omegaconf import OmegaConf
            config_dict = OmegaConf.to_container(args, resolve=True)
        else:
            # 이미 dict인 경우
            config_dict = dict(args)

        # 직렬화 불가능한 객체 제거/변환
        def make_serializable(obj):
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            else:
                return str(obj)

        config_dict = make_serializable(config_dict)

        # 저장 시간 추가
        config_dict['_saved_at'] = datetime.now(timezone(timedelta(hours=9))).isoformat()

        with open(self._config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"Config saved to: {self._config_path}")

    def _write_header_if_needed(self):
        """헤더가 아직 안 쓰여졌으면 작성"""
        if not self._header_written:
            with open(self._save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=UNIFIED_COLUMNS)
                writer.writeheader()
            self._header_written = True

    def _append_row(self, row: Dict[str, Any]):
        """CSV에 한 row append"""
        self._write_header_if_needed()
        with open(self._save_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=UNIFIED_COLUMNS, extrasaction="ignore")
            writer.writerow(row)
        self.sample_count += 1

    def add_classification_sample(
        self,
        idx: int,
        task: str,
        label: str,
        pred: int,
        prob: float,
        correct: int,
    ):
        """Classification 태스크 샘플 추가 및 즉시 저장"""
        row = {
            "idx": idx,
            "task": task,
            "label": label,
            "pred": pred,
            "metakey": CLASSIFICATION_METAKEYS,
            "prob": prob,
            "correct": correct,
        }
        self._append_row(row)

    def add_regression_sample(
        self,
        idx: int,
        task: str,
        label: str,
        pred: str,  # 원본 또는 파싱된 문자열 그대로 저장
        error: float,
        parse_failed: int = 0,  # 1이면 파싱 실패 (offline 평가에서 제외)
    ):
        """Regression 태스크 샘플 추가 및 즉시 저장"""
        row = {
            "idx": idx,
            "task": task,
            "label": label,
            "pred": pred,
            "metakey": REGRESSION_METAKEYS,
            "error": error,
            "parse_failed": parse_failed,
        }
        self._append_row(row)

    def add_molecule_generation_sample(
        self,
        idx: int,
        task: str,
        label: str,
        pred: str,
        validity: int,
        exact_match: int,
        MACCS_FTS: Optional[float] = None,
        RDK_FTS: Optional[float] = None,
        morgan_FTS: Optional[float] = None,
        levenshtein: Optional[int] = None,
    ):
        """Molecule Generation 태스크 샘플 추가 및 즉시 저장"""
        row = {
            "idx": idx,
            "task": task,
            "label": label,
            "pred": pred,
            "metakey": MOLECULE_GENERATION_METAKEYS,
            "validity": validity,
            "exact_match": exact_match,
            "MACCS_FTS": MACCS_FTS,
            "RDK_FTS": RDK_FTS,
            "morgan_FTS": morgan_FTS,
            "levenshtein": levenshtein,
        }
        self._append_row(row)

    def add_captioning_sample(
        self,
        idx: int,
        task: str,
        label: str,
        pred: str,
    ):
        """Captioning 태스크 샘플 추가 및 즉시 저장"""
        row = {
            "idx": idx,
            "task": task,
            "label": label,
            "pred": pred,
            "metakey": CAPTIONING_METAKEYS,
        }
        self._append_row(row)

    def add_sample(self, task: str, **kwargs):
        """범용 샘플 추가 메서드 - task type에 따라 자동 분류"""
        task_type = get_task_type(task)

        if task_type == "classification":
            row = {"task": task, "metakey": CLASSIFICATION_METAKEYS, **kwargs}
        elif task_type == "regression":
            row = {"task": task, "metakey": REGRESSION_METAKEYS, **kwargs}
        elif task_type == "molecule_generation":
            row = {"task": task, "metakey": MOLECULE_GENERATION_METAKEYS, **kwargs}
        elif task_type == "captioning":
            row = {"task": task, "metakey": CAPTIONING_METAKEYS, **kwargs}
        else:
            print(f"[ResultSaver] Unknown task type for task: {task}")
            return

        self._append_row(row)

    def __len__(self) -> int:
        return self.sample_count

    @property
    def dir_path(self) -> str:
        """저장 디렉토리 경로 반환"""
        return self._dir_path

    @property
    def save_path(self) -> str:
        """저장 파일 경로 반환"""
        return self._save_path

    def finalize(self) -> str:
        """저장 완료 메시지 출력 및 경로 반환"""
        if self.sample_count == 0:
            print(f"[ResultSaver] No results saved")
        else:
            print(f"[ResultSaver] Total {self.sample_count} samples saved to {self._save_path}")
        return self._save_path

    # 하위 호환성을 위한 save() 메서드
    def save(self) -> Dict[str, Optional[str]]:
        """하위 호환성을 위한 메서드 - 이미 실시간으로 저장되므로 경로만 반환"""
        self.finalize()
        return {"unified": self._save_path}

    def process_and_save_all(
        self,
        list_logs: Dict[str, List],
        converted_predictions: List[str],
    ) -> Dict[str, Optional[str]]:
        """
        list_logs 데이터를 처리하여 통합 CSV로 저장

        Args:
            list_logs: evaluation에서 수집된 데이터
                - idx: 샘플 인덱스 리스트
                - tasks: task 이름 리스트
                - targets: target 문자열 리스트
                - predictions: raw prediction 리스트
                - probs: probability 리스트 (classification용)
            converted_predictions: per_device_evaluate에서 변환된 predictions

        Returns:
            저장 경로 딕셔너리
        """
        import re
        try:
            import selfies
            from rdkit import Chem
            from rdkit.Chem import MACCSkeys, AllChem
            from rdkit import DataStructs
            from Levenshtein import distance as lev
            HAS_RDKIT = True
        except ImportError:
            HAS_RDKIT = False
            print("[ResultSaver] Warning: rdkit not available, molecule metrics will be skipped")

        num_samples = len(list_logs.get("tasks", []))

        for i in range(num_samples):
            idx = list_logs["idx"][i] if i < len(list_logs.get("idx", [])) else i
            task = list_logs["tasks"][i]
            target = list_logs["targets"][i]
            # converted_predictions 사용 (온라인 평가에서 실제로 사용된 값)
            pred_raw = converted_predictions[i] if i < len(converted_predictions) else list_logs["predictions"][i]
            prob = list_logs["probs"][i] if i < len(list_logs.get("probs", [])) else None

            task_base = task.split("/")[0]
            task_type = get_task_type(task_base)

            if task_type == "classification":
                self._process_classification_sample(idx, task_base, target, prob)

            elif task_type == "regression":
                self._process_regression_sample(idx, task_base, target, pred_raw)

            elif task_type == "molecule_generation":
                self._process_molecule_generation_sample(
                    idx, task_base, target, pred_raw, HAS_RDKIT
                )

            elif task_type == "captioning":
                self._process_captioning_sample(idx, task_base, target, pred_raw)

        return self.save()

    def _process_classification_sample(
        self, idx: int, task: str, target: str, prob: Optional[List[float]]
    ):
        """Classification 샘플 처리"""
        label_val = 1 if ("True" in target or "true" in target) else 0

        if prob and len(prob) >= 2 and prob[0] >= 0:
            pred_val = 1 if prob[1] > prob[0] else 0
            prob_val = prob[1]
        else:
            pred_val = -1
            prob_val = -1.0

        correct = int(pred_val == label_val) if pred_val >= 0 else 0

        self.add_classification_sample(
            idx=idx, task=task, label=target,
            pred=pred_val, prob=prob_val, correct=correct,
        )

    def _process_regression_sample(
        self, idx: int, task: str, target: str, pred_raw: str
    ):
        """Regression 샘플 처리 - converted_predictions (온라인 평가 결과) 그대로 저장"""
        import re

        # Parse target for error calculation
        label_val = None
        match = re.search(r"(?<=<FLOAT>).*?(?=</FLOAT>)", target)
        if match:
            inner = match.group().replace(" ", "").replace("<|", "").replace("|>", "")
            try:
                label_val = float(inner)
            except:
                pass

        # pred_raw는 converted_predictions에서 온 값
        # - 온라인 평가 성공 시: 파싱된 숫자 문자열 (예: "-0.94")
        # - 온라인 평가 실패 시: 원본 그대로 (파싱 불가)
        parse_failed = 0
        error = float('nan')

        try:
            pred_val = float(pred_raw)
            # 파싱 성공 → error 계산
            if label_val is not None:
                error = pred_val - label_val
        except:
            # 파싱 실패 → 원본 그대로 저장, parse_failed flag 설정
            parse_failed = 1

        self.add_regression_sample(
            idx=idx, task=task, label=target,
            pred=pred_raw,  # 원본 그대로 저장
            error=error,
            parse_failed=parse_failed,
        )

    def _process_molecule_generation_sample(
        self, idx: int, task: str, target: str, pred_raw: str, has_rdkit: bool
    ):
        """Molecule Generation 샘플 처리"""
        import re

        validity = 0
        exact_match = 0
        maccs_fts = None
        rdk_fts = None
        morgan_fts = None
        levenshtein = None
        pred_smiles = pred_raw

        if has_rdkit:
            import selfies
            from rdkit import Chem
            from rdkit.Chem import MACCSkeys, AllChem
            from rdkit import DataStructs
            from Levenshtein import distance as lev

            # Parse target SELFIES -> SMILES
            target_selfies_match = re.search(
                r"(?<=<SELFIES>).*?(?=</SELFIES>)", target.replace(" ", "")
            )
            if not target_selfies_match:
                target_selfies_match = re.search(
                    r"(?<=<SELFIES>).*", target.replace(" ", "")
                )

            target_smiles = None
            target_mol = None
            if target_selfies_match:
                try:
                    target_selfies = target_selfies_match.group()
                    target_smiles = selfies.decoder(target_selfies)
                    target_mol = Chem.MolFromSmiles(target_smiles)
                except:
                    pass

            # Parse prediction
            pred_mol = None
            parsed_mol = _parse_flexible_molecule(pred_raw)
            if parsed_mol:
                try:
                    # SELFIES인지 SMILES인지 판단
                    if parsed_mol.startswith('[') and '][' in parsed_mol:
                        pred_smiles = selfies.decoder(parsed_mol)
                    else:
                        pred_smiles = parsed_mol
                    pred_mol = Chem.MolFromSmiles(pred_smiles)
                except:
                    pred_smiles = pred_raw

            if pred_mol is not None:
                validity = 1
                try:
                    pred_canonical = Chem.CanonSmiles(pred_smiles)
                except:
                    pred_canonical = pred_smiles

                if target_mol is not None:
                    try:
                        target_canonical = Chem.CanonSmiles(target_smiles)
                        exact_match = int(
                            Chem.MolToInchi(pred_mol) == Chem.MolToInchi(target_mol)
                        )
                        maccs_fts = DataStructs.FingerprintSimilarity(
                            MACCSkeys.GenMACCSKeys(target_mol),
                            MACCSkeys.GenMACCSKeys(pred_mol),
                            metric=DataStructs.TanimotoSimilarity,
                        )
                        rdk_fts = DataStructs.FingerprintSimilarity(
                            Chem.RDKFingerprint(target_mol),
                            Chem.RDKFingerprint(pred_mol),
                            metric=DataStructs.TanimotoSimilarity,
                        )
                        morgan_fts = DataStructs.TanimotoSimilarity(
                            AllChem.GetMorganFingerprint(target_mol, 2),
                            AllChem.GetMorganFingerprint(pred_mol, 2),
                        )
                        levenshtein = lev(target_canonical, pred_canonical)
                    except:
                        pass

        self.add_molecule_generation_sample(
            idx=idx, task=task, label=target,
            pred=pred_smiles,
            validity=validity, exact_match=exact_match,
            MACCS_FTS=maccs_fts, RDK_FTS=rdk_fts,
            morgan_FTS=morgan_fts, levenshtein=levenshtein,
        )

    def _process_captioning_sample(
        self, idx: int, task: str, target: str, pred_raw: str
    ):
        """Captioning 샘플 처리"""
        pred_text = _parse_flexible_caption(pred_raw)
        if pred_text is None:
            pred_text = pred_raw

        self.add_captioning_sample(
            idx=idx, task=task, label=target, pred=pred_text,
        )


# ============ Helper functions for parsing ============

def _parse_flexible_number(text: str) -> Optional[float]:
    """숫자 파싱 (regression용)"""
    import re
    if not text:
        return None

    # EOS 토큰 제거
    for eos_token in ['</s>', '<|end_of_text|>', '<|endoftext|>', '<|eot_id|>']:
        text = text.replace(eos_token, '')

    # <FLOAT>...</FLOAT> 형식
    match = re.search(r'<FLOAT>\s*(.*?)\s*</FLOAT>', text)
    if match:
        inner = match.group(1).replace(' ', '').replace('<|', '').replace('|>', '')
        try:
            return float(inner)
        except:
            pass

    # <NUMBER>...</NUMBER> 형식
    match = re.search(r'<NUMBER>\s*(.*?)\s*</NUMBER>', text)
    if match:
        try:
            return float(match.group(1).strip())
        except:
            pass

    # 숫자 직접 추출
    text_stripped = text.strip()
    match = re.match(r'^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', text_stripped)
    if match:
        try:
            return float(match.group(1))
        except:
            pass

    return None


def _parse_flexible_molecule(text: str) -> Optional[str]:
    """분자 파싱 (molecule generation용)"""
    import re
    if not text:
        return None

    # EOS 토큰 제거
    for eos_token in ['</s>', '<|eot_id|>']:
        if eos_token in text:
            text = text.split(eos_token)[0]

    # <SELFIES>...</SELFIES> 형식
    match = re.search(r'<SELFIES>\s*(.*?)\s*</SELFIES>', text)
    if match:
        return match.group(1).replace(' ', '')

    # <SMILES>...</SMILES> 형식
    match = re.search(r'<SMILES>\s*(.*?)\s*</SMILES>', text)
    if match:
        return match.group(1).strip()

    # [START_I_SMILES]...[END_I_SMILES] 형식 (Galactica)
    match = re.search(r'\[START_I_SMILES\](.*?)\[END_I_SMILES\]', text)
    if match:
        return match.group(1).strip()

    # [END_I_SMILES]로 끝나는 경우
    match = re.search(r'^(.*?)\[END_I_SMILES\]', text)
    if match:
        return match.group(1).strip()

    return None


def _parse_flexible_caption(text: str) -> Optional[str]:
    """캡션 파싱 (captioning용)"""
    import re
    if not text:
        return None

    # EOS 토큰 앞에서 자르기 (replace가 아니라 split)
    for eos_token in ['</s>', '<|end_of_text|>', '<|endoftext|>', '<|eot_id|>']:
        if eos_token in text:
            text = text.split(eos_token)[0]

    # <DESCRIPTION>...</DESCRIPTION> 형식
    match = re.search(r'<DESCRIPTION>\s*(.*?)\s*</DESCRIPTION>', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 특수 토큰 제거 후 반환
    text = re.sub(r'<[^>]+>', '', text)
    text = text.strip()

    return text if text else None


def merge_rank_results(
    base_dir: str,
    date_str: str,
    script_name: str,
) -> str:
    """
    여러 rank의 통합 결과 파일들을 하나로 병합

    Args:
        base_dir: 결과 저장 기본 경로
        date_str: 날짜 문자열 (e.g., "20251216")
        script_name: 스크립트 이름 (e.g., "llasmol_test")

    Returns:
        병합된 파일 경로
    """
    import glob
    import pandas as pd

    dir_path = os.path.join(base_dir, date_str)
    pattern = os.path.join(dir_path, f"*_{script_name}_rank*.csv")

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")

    dfs = [pd.read_csv(f) for f in files]
    merged = pd.concat(dfs, ignore_index=True)

    # 중복 제거 (idx 기준)
    merged = merged.drop_duplicates(subset=["idx"], keep="first")

    # idx 기준 정렬
    merged = merged.sort_values("idx").reset_index(drop=True)

    # 병합 파일 저장
    time_str = os.path.basename(files[0]).split("_")[0]
    merged_path = os.path.join(dir_path, f"{time_str}_{script_name}_merged.csv")
    merged.to_csv(merged_path, index=False)

    print(f"[ResultSaver] Merged {len(files)} files ({len(merged)} samples) to {merged_path}")
    return merged_path
