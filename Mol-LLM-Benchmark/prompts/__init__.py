"""
Prompt Templates and Formatters for Mol-LLM-Benchmark

프롬프트 템플릿 및 포맷터 모듈.

Templates:
- chemdfm: ChemDFM 스타일 (expert chemist instruction format)
- llasmol: LlaSMol/SMolInstruct 스타일 (natural question format with <SMILES> tags)
- defaults: 기본 fallback 프롬프트 (task type별)

Formatter:
- PromptFormatter: 모델별 프롬프트 포맷팅 클래스

Constants:
- CLASSIFICATION_BENCHMARKS, REGRESSION_BENCHMARKS, etc.
- get_task_type, task2id, id2task
"""

# Templates
from .chemdfm import CHEMDFM_PROMPTS
from .llasmol import LLASMOL_PROMPTS
from .defaults import DEFAULT_PROMPTS

# Constants
from .constants import (
    CLASSIFICATION_BENCHMARKS,
    REGRESSION_BENCHMARKS,
    REACTION_BENCHMARKS,
    TEXT2MOL_BENCHMARKS,
    MOL2TEXT_BENCHMARKS,
    NAME_CONVERSION_BENCHMARKS,
    ALL_TASKS,
    get_task_type,
    task2id,
    id2task,
)

# Formatter
from .formatter import (
    PromptFormatter,
    format_prompt_for_galactica,
    format_prompt_for_llama,
    format_prompt_for_mistral,
    format_prompt_for_gpt,
    format_prompt_for_llasmol,
    format_prompt_for_chemdfm,
    format_prompt_for_3d_molm,
    format_prompt_for_mol_llm,
    selfies_to_smiles,
    extract_description_from_prompt,
)

__all__ = [
    # Templates
    "CHEMDFM_PROMPTS",
    "LLASMOL_PROMPTS",
    "DEFAULT_PROMPTS",
    # Constants
    "CLASSIFICATION_BENCHMARKS",
    "REGRESSION_BENCHMARKS",
    "REACTION_BENCHMARKS",
    "TEXT2MOL_BENCHMARKS",
    "MOL2TEXT_BENCHMARKS",
    "NAME_CONVERSION_BENCHMARKS",
    "ALL_TASKS",
    "get_task_type",
    "task2id",
    "id2task",
    # Formatter
    "PromptFormatter",
    "format_prompt_for_galactica",
    "format_prompt_for_llama",
    "format_prompt_for_mistral",
    "format_prompt_for_gpt",
    "format_prompt_for_llasmol",
    "format_prompt_for_chemdfm",
    "format_prompt_for_3d_molm",
    "selfies_to_smiles",
    "extract_description_from_prompt",
]
