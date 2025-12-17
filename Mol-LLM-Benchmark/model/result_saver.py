"""
ResultSaver: 샘플별 평가 결과를 Task Type별 CSV로 저장하는 클래스

저장 경로: /workspace/Mol_DA_repo/Mol-LLM-Benchmark/results/{날짜}/{시간}_{스크립트명}_{task_type}.csv
시간대: KST (UTC+9)
"""

import os
import csv
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List


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


# Task type별 CSV 컬럼 정의
CLASSIFICATION_COLUMNS = ["idx", "task", "label", "pred", "prob", "correct"]
REGRESSION_COLUMNS = ["idx", "task", "label", "pred", "error"]
MOLECULE_GENERATION_COLUMNS = [
    "idx", "task", "label", "pred", "validity", "exact_match",
    "MACCS_FTS", "RDK_FTS", "morgan_FTS", "levenshtein"
]
CAPTIONING_COLUMNS = ["idx", "task", "label", "pred"]


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
    """샘플별 평가 결과를 Task Type별 CSV로 저장하는 클래스"""

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

        # Task type별 결과 저장
        self.classification_results: List[Dict[str, Any]] = []
        self.regression_results: List[Dict[str, Any]] = []
        self.molecule_generation_results: List[Dict[str, Any]] = []
        self.captioning_results: List[Dict[str, Any]] = []

        # 저장 경로 초기화
        self._date_str: str = ""
        self._time_str: str = ""
        self._dir_path: str = ""
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

    def _get_save_path(self, task_type: str) -> str:
        """Task type별 저장 경로 반환"""
        filename = f"{self._time_str}_{self.script_name}_{task_type}_rank{self.rank}.csv"
        return os.path.join(self._dir_path, filename)

    def add_classification_sample(
        self,
        idx: int,
        task: str,
        label: str,
        pred: int,
        prob: float,
        correct: int,
    ):
        """Classification 태스크 샘플 추가"""
        self.classification_results.append({
            "idx": idx,
            "task": task,
            "label": label,
            "pred": pred,
            "prob": prob,
            "correct": correct,
        })

    def add_regression_sample(
        self,
        idx: int,
        task: str,
        label: str,
        pred: float,
        error: float,
    ):
        """Regression 태스크 샘플 추가"""
        self.regression_results.append({
            "idx": idx,
            "task": task,
            "label": label,
            "pred": pred,
            "error": error,
        })

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
        """Molecule Generation 태스크 샘플 추가"""
        self.molecule_generation_results.append({
            "idx": idx,
            "task": task,
            "label": label,
            "pred": pred,
            "validity": validity,
            "exact_match": exact_match,
            "MACCS_FTS": MACCS_FTS,
            "RDK_FTS": RDK_FTS,
            "morgan_FTS": morgan_FTS,
            "levenshtein": levenshtein,
        })

    def add_captioning_sample(
        self,
        idx: int,
        task: str,
        label: str,
        pred: str,
    ):
        """Captioning 태스크 샘플 추가"""
        self.captioning_results.append({
            "idx": idx,
            "task": task,
            "label": label,
            "pred": pred,
        })

    def add_sample(self, task: str, **kwargs):
        """범용 샘플 추가 메서드 - task type에 따라 자동 분류"""
        task_type = get_task_type(task)

        if task_type == "classification":
            self.classification_results.append({"task": task, **kwargs})
        elif task_type == "regression":
            self.regression_results.append({"task": task, **kwargs})
        elif task_type == "molecule_generation":
            self.molecule_generation_results.append({"task": task, **kwargs})
        elif task_type == "captioning":
            self.captioning_results.append({"task": task, **kwargs})
        else:
            print(f"[ResultSaver] Unknown task type for task: {task}")

    def _save_csv(self, results: List[Dict], columns: List[str], task_type: str) -> Optional[str]:
        """특정 task type의 결과를 CSV로 저장"""
        if not results:
            return None

        save_path = self._get_save_path(task_type)
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

        print(f"[ResultSaver] Saved {len(results)} {task_type} samples to {save_path}")
        return save_path

    def save(self) -> Dict[str, Optional[str]]:
        """모든 결과를 Task Type별 CSV로 저장하고 저장 경로들 반환"""
        saved_paths = {}

        saved_paths["classification"] = self._save_csv(
            self.classification_results, CLASSIFICATION_COLUMNS, "classification"
        )
        saved_paths["regression"] = self._save_csv(
            self.regression_results, REGRESSION_COLUMNS, "regression"
        )
        saved_paths["molecule_generation"] = self._save_csv(
            self.molecule_generation_results, MOLECULE_GENERATION_COLUMNS, "molecule_generation"
        )
        saved_paths["captioning"] = self._save_csv(
            self.captioning_results, CAPTIONING_COLUMNS, "captioning"
        )

        total = (
            len(self.classification_results) +
            len(self.regression_results) +
            len(self.molecule_generation_results) +
            len(self.captioning_results)
        )

        if total == 0:
            print(f"[ResultSaver] No results to save")
        else:
            print(f"[ResultSaver] Total {total} samples saved")

        return saved_paths

    def __len__(self) -> int:
        return (
            len(self.classification_results) +
            len(self.regression_results) +
            len(self.molecule_generation_results) +
            len(self.captioning_results)
        )

    @property
    def dir_path(self) -> str:
        """저장 디렉토리 경로 반환"""
        return self._dir_path


def merge_rank_results(
    base_dir: str,
    date_str: str,
    script_name: str,
    task_type: str,
) -> str:
    """
    여러 rank의 결과 파일들을 하나로 병합

    Args:
        base_dir: 결과 저장 기본 경로
        date_str: 날짜 문자열 (e.g., "20251216")
        script_name: 스크립트 이름 (e.g., "llasmol_test")
        task_type: 태스크 유형 (classification, regression, molecule_generation, captioning)

    Returns:
        병합된 파일 경로
    """
    import glob
    import pandas as pd

    dir_path = os.path.join(base_dir, date_str)
    pattern = os.path.join(dir_path, f"*_{script_name}_{task_type}_rank*.csv")

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
    merged_path = os.path.join(dir_path, f"{time_str}_{script_name}_{task_type}_merged.csv")
    merged.to_csv(merged_path, index=False)

    print(f"[ResultSaver] Merged {len(files)} files ({len(merged)} samples) to {merged_path}")
    return merged_path


def merge_all_rank_results(
    base_dir: str,
    date_str: str,
    script_name: str,
) -> Dict[str, str]:
    """
    모든 task type에 대해 rank 결과 병합

    Args:
        base_dir: 결과 저장 기본 경로
        date_str: 날짜 문자열
        script_name: 스크립트 이름

    Returns:
        task_type -> 병합된 파일 경로 딕셔너리
    """
    merged_paths = {}
    task_types = ["classification", "regression", "molecule_generation", "captioning"]

    for task_type in task_types:
        try:
            merged_paths[task_type] = merge_rank_results(
                base_dir, date_str, script_name, task_type
            )
        except FileNotFoundError:
            print(f"[ResultSaver] No {task_type} files to merge")
            merged_paths[task_type] = None

    return merged_paths
