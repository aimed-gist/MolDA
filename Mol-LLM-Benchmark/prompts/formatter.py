"""
Prompt Formatter 클래스

각 모델별 프롬프트 포맷팅 로직을 담당.
"""

import re
from typing import Optional

import selfies as sf

from .chemdfm import CHEMDFM_PROMPTS
from .llasmol import LLASMOL_PROMPTS
from .defaults import DEFAULT_PROMPTS
from .constants import get_task_type


def selfies_to_smiles(selfies_str: str) -> Optional[str]:
    """SELFIES → SMILES (canonicalized)"""
    try:
        from rdkit import Chem
        smiles = sf.decoder(selfies_str)
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol)
        return smiles
    except:
        return None


def extract_description_from_prompt(prompt_text: str) -> Optional[str]:
    """Text2Mol task에서 description 추출"""
    match = re.search(r'<DESCRIPTION>\s*(.*?)\s*</DESCRIPTION>', prompt_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def convert_all_selfies_tags(text: str, converter_func=None) -> str:
    """
    텍스트 내의 모든 <SELFIES>...</SELFIES> 태그를 SMILES로 변환

    Args:
        text: 변환할 텍스트
        converter_func: SMILES 변환 후 추가 포맷팅 함수 (예: Galactica의 [START_I_SMILES]...[END_I_SMILES])
                       None이면 순수 SMILES만 반환

    Returns:
        모든 SELFIES 태그가 SMILES로 변환된 텍스트
    """
    result = text
    while '<SELFIES>' in result and '</SELFIES>' in result:
        start_idx = result.find('<SELFIES>')
        end_idx = result.find('</SELFIES>') + len('</SELFIES>')
        selfies_content = result[start_idx + len('<SELFIES>'):end_idx - len('</SELFIES>')].strip()

        # SELFIES를 SMILES로 변환
        converted_smiles = selfies_to_smiles(selfies_content)
        if converted_smiles is None:
            try:
                converted_smiles = sf.decoder(selfies_content)
            except:
                converted_smiles = selfies_content

        # 추가 포맷팅 적용 (예: Galactica 형식)
        if converter_func:
            converted_smiles = converter_func(converted_smiles)

        result = result[:start_idx] + converted_smiles + result[end_idx:]

    return result


def remove_graph_tags(text: str) -> str:
    """<GRAPH>...</GRAPH> 태그 및 내용 제거"""
    return re.sub(r'<GRAPH>.*?</GRAPH>', '', text, flags=re.DOTALL)


def remove_startoftext_tag(text: str) -> str:
    """<|startoftext|> 태그 제거"""
    return text.replace('<|startoftext|>', '')


def remove_llama3_tags(text: str) -> str:
    """Llama3 형식 태그 제거"""
    result = text
    # <|start_header_id|>system<|end_header_id|> 제거
    result = result.replace('<|start_header_id|>system<|end_header_id|>\n\n', '')
    result = result.replace('<|start_header_id|>system<|end_header_id|>', '')
    # <|eot_id|><|start_header_id|>user<|end_header_id|> 제거
    result = result.replace('<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n', '\n\n')
    result = result.replace('<|eot_id|><|start_header_id|>user<|end_header_id|>', '\n\n')
    # <|eot_id|><|start_header_id|>assistant<|end_header_id|> 제거
    result = result.replace('<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n', '')
    result = result.replace('<|eot_id|><|start_header_id|>assistant<|end_header_id|>', '')
    # 단독 <|eot_id|> 제거
    result = result.replace('<|eot_id|>', '')
    return result


def remove_mistral_tags(text: str) -> str:
    """Mistral 형식 태그 제거"""
    result = re.sub(r'^<s>\s*', '', text)
    result = result.replace('[INST]', '')
    result = result.replace('[/INST]', '')
    return result


def clean_whitespace(text: str) -> str:
    """공백 정리 (3개 이상의 연속 줄바꿈을 2개로)"""
    result = re.sub(r'\n{3,}', '\n\n', text)
    return result.strip()


def preprocess_prompt(
    text: str,
    selfies_converter=None,
    remove_llama3: bool = True,
    remove_mistral: bool = False,
    replace_selfies_text: bool = True,
) -> str:
    """
    프롬프트 전처리 공통 함수

    Args:
        text: 원본 프롬프트
        selfies_converter: SELFIES→SMILES 변환 후 추가 포맷팅 함수
                          None이면 순수 SMILES만 반환
                          예: lambda s: f'[START_I_SMILES]{s}[END_I_SMILES]' (Galactica)
        remove_llama3: Llama3 형식 태그 제거 여부
        remove_mistral: Mistral 형식 태그 제거 여부
        replace_selfies_text: "SELFIES" 텍스트를 "SMILES"로 변환 여부

    Returns:
        전처리된 프롬프트
    """
    result = text

    # 1. 공통 태그 제거
    result = remove_startoftext_tag(result)
    result = remove_graph_tags(result)

    # 2. <SELFIES>...</SELFIES> 태그를 SMILES로 변환
    result = convert_all_selfies_tags(result, converter_func=selfies_converter)

    # 3. 모델별 태그 제거
    if remove_llama3:
        result = remove_llama3_tags(result)
    if remove_mistral:
        result = remove_mistral_tags(result)

    # 4. "SELFIES" 텍스트 → "SMILES"로 변환
    if replace_selfies_text:
        result = result.replace('SELFIES', 'SMILES')

    # 5. 공백 정리
    result = clean_whitespace(result)

    return result


class PromptFormatter:
    """
    모델별 프롬프트 포맷터

    Usage:
        formatter = PromptFormatter("galactica")
        formatted = formatter.format(prompt, selfies_str, task_name)
    """

    SUPPORTED_MODELS = ["galactica", "llama", "mistral", "gpt", "llasmol", "chemdfm"]

    def __init__(self, model_name: str):
        """
        Args:
            model_name: 모델 이름 (galactica, llama, mistral, gpt, llasmol, chemdfm)
        """
        self.model_name = model_name.lower()
        if self.model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported: {self.SUPPORTED_MODELS}")

    def format(self, prompt: str, selfies_str: str, task_name: str) -> str:
        """
        프롬프트를 모델에 맞게 포맷팅

        Args:
            prompt: 원본 프롬프트
            selfies_str: SELFIES 분자 표현
            task_name: Task 이름

        Returns:
            포맷팅된 프롬프트
        """
        if self.model_name == "galactica":
            return self._format_galactica(prompt, selfies_str, task_name)
        elif self.model_name == "llama":
            return self._format_llama(prompt, selfies_str, task_name)
        elif self.model_name == "mistral":
            return self._format_mistral(prompt, selfies_str, task_name)
        elif self.model_name == "gpt":
            return self._format_gpt(prompt, selfies_str, task_name)
        elif self.model_name == "llasmol":
            return self._format_llasmol(prompt, selfies_str, task_name)
        elif self.model_name == "chemdfm":
            return self._format_chemdfm(prompt, selfies_str, task_name)
        else:
            return prompt

    def _format_galactica(self, prompt: str, selfies_str: str, task_name: str) -> str:
        """
        Galactica 모델용 프롬프트 포맷팅
        - SELFIES → SMILES 변환
        - [START_I_SMILES]...[END_I_SMILES] 포맷 적용
        """
        smiles_str = selfies_to_smiles(selfies_str)
        if smiles_str is None:
            try:
                smiles_str = sf.decoder(selfies_str)
            except:
                smiles_str = selfies_str

        formatted = prompt

        # 1. 공통 태그 제거
        formatted = remove_startoftext_tag(formatted)
        formatted = remove_graph_tags(formatted)

        # 2. <SELFIES>...</SELFIES> 태그를 Galactica SMILES 포맷으로 변환 (모든 태그 처리)
        galactica_converter = lambda s: f'[START_I_SMILES]{s}[END_I_SMILES]'
        formatted = convert_all_selfies_tags(formatted, converter_func=galactica_converter)

        # 3. Llama3/Mistral 형식 태그 제거
        formatted = remove_llama3_tags(formatted)
        formatted = remove_mistral_tags(formatted)

        # 4. "SELFIES" → "SMILES"
        formatted = formatted.replace('SELFIES', 'SMILES')

        # 5. 공백 정리
        formatted = re.sub(r'\n{2,}', '\n', formatted)
        formatted = formatted.strip()

        # 8. Task별 Answer hint 추가
        task_type = get_task_type(task_name)
        if task_type == "classification":
            formatted += "\nAnswer (True or False):"
        elif task_type == "regression":
            formatted += "\nAnswer with a number only:"
        elif task_type == "reaction":
            formatted += "\nAnswer with SMILES only. Product: [START_I_SMILES]"
        elif task_type == "mol2text":
            formatted += "\nExplanation :"
        elif task_type == "text2mol":
            formatted += "\nAnswer with SMILES only: [START_I_SMILES]"

        return formatted.rstrip()

    def _format_llama(self, prompt: str, selfies_str: str, task_name: str) -> str:
        """Llama 모델용 (원본 유지)"""
        return prompt

    def _format_mistral(self, prompt: str, selfies_str: str, task_name: str) -> str:
        """Mistral 모델용 (원본 유지)"""
        return prompt

    def _format_gpt(self, prompt: str, selfies_str: str, task_name: str) -> str:
        """GPT 모델용 - instruction wrapper 제거"""
        prompt = re.sub(r'<s>\[INST\]\s*', '', prompt)
        prompt = prompt.replace('[/INST]', '')
        return prompt.strip()

    def _format_llasmol(self, prompt: str, selfies_str: str, task_name: str) -> str:
        """
        LlaSMol 모델용
        - SELFIES → SMILES 변환
        - <SMILES>...</SMILES> 태그 사용
        - [INST] wrapper는 LlaSMolGeneration이 내부적으로 처리
        """
        smiles_str = selfies_to_smiles(selfies_str)
        if smiles_str is None:
            try:
                smiles_str = sf.decoder(selfies_str)
            except:
                smiles_str = selfies_str

        task_base = task_name.split("/")[0]
        description = extract_description_from_prompt(prompt)

        # LlaSMol 템플릿 사용
        if task_base not in LLASMOL_PROMPTS:
            # Fallback to default
            task_type = get_task_type(task_name)
            template = DEFAULT_PROMPTS.get(task_type, DEFAULT_PROMPTS["regression"])
        else:
            template = LLASMOL_PROMPTS[task_base]

        if "{description}" in template and description:
            formatted = template.format(description=description)
        elif "{smiles}" in template:
            formatted = template.format(smiles=smiles_str)
        else:
            formatted = template

        return formatted

    def _format_chemdfm(self, prompt: str, selfies_str: str, task_name: str) -> str:
        """
        ChemDFM 모델용
        - ChemDFM 공식 프롬프트 스타일 사용
        - Format: [Round 0]\nHuman: ...\nAssistant:
        """
        smiles_str = selfies_to_smiles(selfies_str)
        if smiles_str is None:
            try:
                smiles_str = sf.decoder(selfies_str)
            except:
                smiles_str = selfies_str

        task_base = task_name.split("/")[0]
        task_type = get_task_type(task_name)
        description = extract_description_from_prompt(prompt)

        # ChemDFM 템플릿 사용
        if task_base in CHEMDFM_PROMPTS:
            template = CHEMDFM_PROMPTS[task_base]
        else:
            template = DEFAULT_PROMPTS.get(task_type, DEFAULT_PROMPTS["regression"])

        if "{description}" in template and description:
            formatted = template.format(description=description)
        elif "{smiles}" in template:
            formatted = template.format(smiles=smiles_str)
        else:
            formatted = template

        # ChemDFM wrapper
        return f"[Round 0]\nHuman: {formatted}\nAssistant:"

    def _format_chemdfm_passthrough(self, prompt: str, selfies_str: str, task_name: str, wrapper: bool = False) -> str:
        """
        ChemDFM 모델용 - Passthrough 모드
        - 원본 프롬프트를 최대한 유지하면서 불필요한 태그만 제거
        - SELFIES → SMILES 변환

        변환 규칙:
        - <|startoftext|> 제거
        - <|start_header_id|>system<|end_header_id|> → 제거 (시스템 프롬프트 내용은 유지)
        - <|eot_id|><|start_header_id|>user<|end_header_id|> → 줄바꿈으로 대체
        - <SELFIES>...</SELFIES> → SMILES로 변환 (태그 제거)
        - <GRAPH>...</GRAPH> → 내용 포함 전체 제거
        - <|eot_id|><|start_header_id|>assistant<|end_header_id|> → 제거
        - "SELFIES" 텍스트 → "SMILES"로 변환

        Args:
            wrapper: True면 [Round 0]\nHuman: ...\nAssistant: 형식으로 감싸기
        """
        formatted = prompt

        # 1. <|startoftext|> 제거
        formatted = formatted.replace('<|startoftext|>', '')

        # 2. <|start_header_id|>system<|end_header_id|> 제거 (시스템 프롬프트 내용은 유지)
        formatted = formatted.replace('<|start_header_id|>system<|end_header_id|>\n\n', '')
        formatted = formatted.replace('<|start_header_id|>system<|end_header_id|>', '')

        # 3. <|eot_id|><|start_header_id|>user<|end_header_id|> → 줄바꿈으로 대체
        formatted = formatted.replace('<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n', '\n\n')
        formatted = formatted.replace('<|eot_id|><|start_header_id|>user<|end_header_id|>', '\n\n')

        # 4. <SELFIES>...</SELFIES> → SMILES로 변환 (태그 없이)
        formatted = convert_all_selfies_tags(formatted)

        # 5. <GRAPH>...</GRAPH> 내용 포함 전체 제거
        formatted = remove_graph_tags(formatted)

        # 6. <|eot_id|><|start_header_id|>assistant<|end_header_id|> 제거
        formatted = formatted.replace('<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n', '')
        formatted = formatted.replace('<|eot_id|><|start_header_id|>assistant<|end_header_id|>', '')

        # 7. 남은 <|eot_id|> 제거
        formatted = formatted.replace('<|eot_id|>', '')

        # 8. "SELFIES" 텍스트 → "SMILES"로 변환
        formatted = formatted.replace('SELFIES', 'SMILES')

        # 9. <DESCRIPTION>...</DESCRIPTION> 태그 제거 (내용은 유지)
        formatted = re.sub(r'<DESCRIPTION>(.*?)</DESCRIPTION>', r'\1', formatted, flags=re.DOTALL)

        # 10. 공백 정리
        formatted = clean_whitespace(formatted)

        # wrapper 옵션에 따라 ChemDFM wrapper 적용
        if wrapper:
            return f"[Round 0]\nHuman: {formatted}\nAssistant:"
        else:
            return formatted
    def _format_llasmol_passthrough(self, prompt: str, selfies_str: str, task_name: str, wrapper: bool = False) -> str:


        return None


# 기존 함수들과의 호환성을 위한 wrapper 함수들
def format_prompt_for_galactica(prompt: str, selfies_str: str, task_name: str) -> str:
    """Galactica용 프롬프트 포맷팅 (호환성 유지)"""
    return PromptFormatter("galactica").format(prompt, selfies_str, task_name)


def format_prompt_for_llama(prompt: str, selfies_str: str, task_name: str) -> str:
    """Llama용 프롬프트 포맷팅 (호환성 유지)"""
    return PromptFormatter("llama").format(prompt, selfies_str, task_name)


def format_prompt_for_mistral(prompt: str, selfies_str: str, task_name: str) -> str:
    """Mistral용 프롬프트 포맷팅 (호환성 유지)"""
    return PromptFormatter("mistral").format(prompt, selfies_str, task_name)


def format_prompt_for_gpt(prompt: str, selfies_str: str, task_name: str) -> str:
    """GPT용 프롬프트 포맷팅 (호환성 유지)"""
    return PromptFormatter("gpt").format(prompt, selfies_str, task_name)


def format_prompt_for_llasmol(prompt: str, selfies_str: str, task_name: str) -> str:
    """LlaSMol용 프롬프트 포맷팅 (호환성 유지)"""
    formatter = PromptFormatter("llasmol")
    if intrinsic_prompt:
        return formatter._format_llasmol(prompt, selfies_str, task_name)
    else:
        return formatter._format_llasmol_passthrough(prompt, selfies_str, task_name, wrapper=wrapper)


def format_prompt_for_chemdfm(prompt: str, selfies_str: str, task_name: str, intrinsic_prompt: bool = True, wrapper: bool = True) -> str:
    """
    ChemDFM용 프롬프트 포맷팅

    Args:
        prompt: 원본 프롬프트
        selfies_str: SELFIES 분자 표현
        task_name: Task 이름
        intrinsic_prompt: True면 ChemDFM 맞춤 프롬프트 사용, False면 태그만 제거하고 원본 유지
        wrapper: True면 [Round 0]\nHuman: ...\nAssistant: 형식으로 감싸기 (intrinsic_prompt=False일 때만 적용)
    """
    formatter = PromptFormatter("chemdfm")
    if intrinsic_prompt:
        return formatter._format_chemdfm(prompt, selfies_str, task_name)
    else:
        return formatter._format_chemdfm_passthrough(prompt, selfies_str, task_name, wrapper=wrapper)


# ============ Test Cases ============
CHEMDFM_TEST_CASES = [
    # BACE Classification
    {
        "task": "bace",
        "raw_prompt": """<|startoftext|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation.<|eot_id|><|start_header_id|>user<|end_header_id|>

Predict the biological activity of the molecule <SELFIES> [C][C][=C][C][=C][C][=C][Ring1][=Branch1][C][=C][C][=C][N][=C][Branch1][C][N][C][Branch2][Ring1][Ring2][C][C][Branch1][C][C][C][=Branch1][C][=O][N][C][C][C][C][O][C][C][Ring1][=Branch1][=C][C][Ring2][Ring1][Ring2][=C][Ring2][Ring1][Branch2] </SELFIES> against BACE-1.<GRAPH> <mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol> </GRAPH><|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        "selfies": "[C][C][=C][C][=C][C][=C][Ring1][=Branch1][C][=C][C][=C][N][=C][Branch1][C][N][C][Branch2][Ring1][Ring2][C][C][Branch1][C][C][C][=Branch1][C][=O][N][C][C][C][C][O][C][C][Ring1][=Branch1][=C][C][Ring2][Ring1][Ring2][=C][Ring2][Ring1][Branch2]",
        "expected_output": """[Round 0]
Human: You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether it can inhibit (True) the Beta-site Amyloid Precursor Protein Cleaving Enzyme 1 (BACE1) or cannot inhibit (False). Please answer with only True or False.
SMILES: Cc1ccccc1-c1ccc2nc(N)c(CC(C)C(=O)NCC3CCOCC3)cc2c1
BACE-1 Inhibit:
Assistant:"""
    },
    {
        "task": "bace",
        "raw_prompt": """<|startoftext|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation.<|eot_id|><|start_header_id|>user<|end_header_id|>

Please provide the biological activity value for this molecule against BACE-1: <SELFIES> [C][C][N][C][=C][C][=C][Branch2][Ring2][#Branch2][C][=C][Branch2][Ring1][=N][C][=Branch1][C][=O][N][C][Branch1][#Branch2][C][C][=C][C][=C][C][=C][Ring1][=Branch1][C@H1][Branch1][C][O][C][NH2+1][C][C][C][Ring1][Ring1][C][=C][Ring2][Ring1][Branch2][Ring2][Ring1][O][N][Branch1][C][C][S][=Branch1][C][=O][=Branch1][C][=O][C][C][Ring2][Ring1][S] </SELFIES>.<GRAPH> <mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol> </GRAPH><|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        "selfies": "[C][C][N][C][=C][C][=C][Branch2][Ring2][#Branch2][C][=C][Branch2][Ring1][=N][C][=Branch1][C][=O][N][C][Branch1][#Branch2][C][C][=C][C][=C][C][=C][Ring1][=Branch1][C@H1][Branch1][C][O][C][NH2+1][C][C][C][Ring1][Ring1][C][=C][Ring2][Ring1][Branch2][Ring2][Ring1][O][N][Branch1][C][C][S][=Branch1][C][=O][=Branch1][C][=O][C][C][Ring2][Ring1][S]",
        "expected_output": """[Round 0]
Human: You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether it can inhibit (True) the Beta-site Amyloid Precursor Protein Cleaving Enzyme 1 (BACE1) or cannot inhibit (False). Please answer with only True or False.
SMILES: CCn1cc2c3c(cc(C(=O)NC(Cc4ccccc4)[C@H](O)C[NH2+]C4CC4)cc31)N(C)S(=O)(=O)CC2
BACE-1 Inhibit:
Assistant:"""
    },
    # Mol2Text (Captioning)
    {
        "task": "chebi-20-mol2text",
        "raw_prompt": """<|startoftext|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation.<|eot_id|><|start_header_id|>user<|end_header_id|>

Describe this molecule: <SELFIES> [C][O][C][=C][C][Branch1][C][O][=C][Branch1][C][C][C][=C][Ring1][Branch2][C][=Branch1][C][=O][C][C@@H1][Branch1][=Branch2][C][=C][C][=C][C][=C][Ring1][=Branch1][O][Ring1][=N] </SELFIES><GRAPH> <mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol> </GRAPH><|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        "selfies": "[C][O][C][=C][C][Branch1][C][O][=C][Branch1][C][C][C][=C][Ring1][Branch2][C][=Branch1][C][=O][C][C@@H1][Branch1][=Branch2][C][=C][C][=C][C][=C][Ring1][=Branch1][O][Ring1][=N]",
        "expected_output": """[Round 0]
Human: You are an expert chemist, your task is to describe molecules.
Given the SMILES string of a molecule, provide a detailed description of its structure, functional groups, and properties.
SMILES: COc1cc(O)c(C)c2c1C(=O)C[C@@H](c1ccccc1)O2
Description:
Assistant:"""
    },
    {
        "task": "chebi-20-mol2text",
        "raw_prompt": """<|startoftext|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation.<|eot_id|><|start_header_id|>user<|end_header_id|>

What can you tell me about this molecule? <SELFIES> [C][C][=C][\\C][/C][=C][Branch1][C][\\C][C][C@@H1][C][=C][Branch1][S][C][C][C@H1][Branch1][=Branch1][C][Branch1][C][C][C][/C][=C][-/Ring2][Ring1][C][C][=Branch1][C][=O][O][Ring1][=C] </SELFIES><GRAPH> <mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol> </GRAPH><|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        "selfies": "[C][C][=C][\\C][/C][=C][Branch1][C][\\C][C][C@@H1][C][=C][Branch1][S][C][C][C@H1][Branch1][=Branch1][C][Branch1][C][C][C][/C][=C][-/Ring2][Ring1][C][C][=Branch1][C][=O][O][Ring1][=C]",
        "expected_output": """[Round 0]
Human: You are an expert chemist, your task is to describe molecules.
Given the SMILES string of a molecule, provide a detailed description of its structure, functional groups, and properties.
SMILES: CC1=C/C/C=C(\\C)C[C@@H]2C=C(CC[C@H](C(C)C)/C=C/1)C(=O)O2
Description:
Assistant:"""
    },
    {
        "task": "chebi-20-mol2text",
        "raw_prompt": """<|startoftext|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation.<|eot_id|><|start_header_id|>user<|end_header_id|>

Could you give me a brief introduction to this molecule? <SELFIES> [C][C][C][C][=Branch1][C][=O][C][C][C] </SELFIES><GRAPH> <mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol> </GRAPH><|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        "selfies": "[C][C][C][C][=Branch1][C][=O][C][C][C]",
        "expected_output": """[Round 0]
Human: You are an expert chemist, your task is to describe molecules.
Given the SMILES string of a molecule, provide a detailed description of its structure, functional groups, and properties.
SMILES: CCCC(=O)CCC
Description:
Assistant:"""
    },
]


def test_chemdfm_formatter():
    """ChemDFM formatter 테스트"""
    print("=" * 60)
    print("Testing ChemDFM Formatter")
    print("=" * 60)

    formatter = PromptFormatter("chemdfm")
    passed = 0
    failed = 0

    for i, tc in enumerate(CHEMDFM_TEST_CASES):
        task = tc["task"]
        raw_prompt = tc["raw_prompt"]
        selfies = tc["selfies"]
        expected = tc["expected_output"]

        # Format the prompt
        result = formatter.format(raw_prompt, selfies, task)

        # Compare (normalize whitespace for comparison)
        result_normalized = result.strip()
        expected_normalized = expected.strip()

        if result_normalized == expected_normalized:
            print(f"✓ Test {i+1} ({task}): PASSED")
            passed += 1
        else:
            print(f"✗ Test {i+1} ({task}): FAILED")
            print(f"  Expected:\n{expected_normalized[:200]}...")
            print(f"  Got:\n{result_normalized[:200]}...")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def test_intrinsic_chemdfm():
    """intrinsic_prompt=True/False 비교 테스트"""
    print("=" * 60)
    print("Testing intrinsic_prompt (True vs False)")
    print("=" * 60)

    # 테스트 케이스
    test_cases = [
        {
            "name": "BACE Classification",
            "task": "bace",
            "raw_prompt": """<|startoftext|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation.<|eot_id|><|start_header_id|>user<|end_header_id|>

Predict the biological activity of the molecule <SELFIES> [C][C][=C][C][=C][C][=C][Ring1][=Branch1][C][=C][C][=C][N][=C][Branch1][C][N][C][Branch2][Ring1][Ring2][C][C][Branch1][C][C][C][=Branch1][C][=O][N][C][C][C][C][O][C][C][Ring1][=Branch1][=C][C][Ring2][Ring1][Ring2][=C][Ring2][Ring1][Branch2] </SELFIES> against BACE-1.<GRAPH> <mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol> </GRAPH><|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            "selfies": "[C][C][=C][C][=C][C][=C][Ring1][=Branch1][C][=C][C][=C][N][=C][Branch1][C][N][C][Branch2][Ring1][Ring2][C][C][Branch1][C][C][C][=Branch1][C][=O][N][C][C][C][C][O][C][C][Ring1][=Branch1][=C][C][Ring2][Ring1][Ring2][=C][Ring2][Ring1][Branch2]",
        },
        {
            "name": "Mol2Text (Captioning)",
            "task": "chebi-20-mol2text",
            "raw_prompt": """<|startoftext|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation.<|eot_id|><|start_header_id|>user<|end_header_id|>

Describe this molecule: <SELFIES> [C][O][C][=C][C][Branch1][C][O][=C][Branch1][C][C][C][=C][Ring1][Branch2][C][=Branch1][C][=O][C][C@@H1][Branch1][=Branch2][C][=C][C][=C][C][=C][Ring1][=Branch1][O][Ring1][=N] </SELFIES><GRAPH> <mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol> </GRAPH><|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            "selfies": "[C][O][C][=C][C][Branch1][C][O][=C][Branch1][C][C][C][=C][Ring1][Branch2][C][=Branch1][C][=O][C][C@@H1][Branch1][=Branch2][C][=C][C][=C][C][=C][Ring1][=Branch1][O][Ring1][=N]",
        },
        {
            "name": "Text2Mol (Generation)",
            "task": "chebi-20-text2mol",
            "raw_prompt": """<|startoftext|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant for molecular chemistry, to address tasks including molecular property classification, molecular property regression, chemical reaction prediction, molecule captioning, molecule generation.<|eot_id|><|start_header_id|>user<|end_header_id|>

Generate a molecule based on this description: The molecule is a member of the class of benzofurans.<GRAPH> <mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol><mol> </GRAPH><|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
            "selfies": "",
        },
    ]

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"[{tc['name']}] task={tc['task']}")
        print(f"{'='*60}")

        # # intrinsic_prompt=True
        # print("\n[intrinsic_prompt=True] (ChemDFM 맞춤 프롬프트)")
        # print("-" * 40)
        # result_true = format_prompt_for_chemdfm(
        #     tc["raw_prompt"], tc["selfies"], tc["task"], intrinsic_prompt=True
        # )
        # print(result_true)
        # raw prompt 
        print("\n[raw prompt] ")
        print("-" * 40)
        print(tc["raw_prompt"])
        # intrinsic_prompt=False
        print("\n[intrinsic_prompt=False] (Passthrough - 태그만 제거)")
        print("-" * 40)
        result_false = format_prompt_for_chemdfm(
            tc["raw_prompt"], tc["selfies"], tc["task"], intrinsic_prompt=False
        )
        print(result_false)


if __name__ == "__main__":
    '''
    
    '''
    import sys

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "intrinsic":
        test_intrinsic_chemdfm()
        sys.exit(0)

    # Run default tests
    # success = test_chemdfm_formatter()

    sys.exit(0 if success else 1)
