"""
Benchmark Task 상수 정의

각 task type별 benchmark 리스트 및 관련 유틸리티.
"""

CLASSIFICATION_BENCHMARKS = [
    "smol-property_prediction-bbbp",
    "smol-property_prediction-clintox",
    "smol-property_prediction-hiv",
    "smol-property_prediction-sider",
    "bace",
    "tox21",
    "toxcast",
]

REGRESSION_BENCHMARKS = [
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

REACTION_BENCHMARKS = [
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
]

TEXT2MOL_BENCHMARKS = [
    "chebi-20-text2mol",
    "smol-molecule_generation",
]

MOL2TEXT_BENCHMARKS = [
    "chebi-20-mol2text",
    "smol-molecule_captioning",
]

NAME_CONVERSION_BENCHMARKS = [
    "smol-name_conversion-i2s",
    "smol-name_conversion-i2f",
    "smol-name_conversion-s2f",
    "smol-name_conversion-s2i",
]

# 전체 tasks 리스트
ALL_TASKS = (
    CLASSIFICATION_BENCHMARKS
    + REGRESSION_BENCHMARKS
    + REACTION_BENCHMARKS
    + TEXT2MOL_BENCHMARKS
    + MOL2TEXT_BENCHMARKS
    + NAME_CONVERSION_BENCHMARKS
)


def get_task_type(task_name: str) -> str:
    """Task 이름으로 task type 반환"""
    task_base = task_name.split("/")[0]

    if task_base in CLASSIFICATION_BENCHMARKS:
        return "classification"
    elif task_base in REGRESSION_BENCHMARKS:
        return "regression"
    elif task_base in TEXT2MOL_BENCHMARKS:
        return "text2mol"
    elif task_base in REACTION_BENCHMARKS:
        return "reaction"
    elif task_base in MOL2TEXT_BENCHMARKS:
        return "mol2text"
    elif task_base in NAME_CONVERSION_BENCHMARKS:
        return "name_conversion"
    else:
        return "unknown"


def task2id(task: str) -> int:
    """Task name to task id"""
    task2id_map = {k: i for i, k in enumerate(ALL_TASKS)}
    return task2id_map[task]


def id2task(task_id: int) -> str:
    """Task id to task name"""
    id2task_map = {i: k for i, k in enumerate(ALL_TASKS)}
    return id2task_map[task_id]
