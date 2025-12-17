import torch
from transformers import DataCollatorForSeq2Seq
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater as GraphCollater

import numpy as np

from collections import Counter

import selfies as sf

import rdkit.Chem as Chem
import re
import copy
from typing import Optional


# ============================================
# ChemDFM 스타일 프롬프트 템플릿 (preprocess_for_external.py 참조)
# ============================================
CHEMDFM_PROMPTS = {
    # ============ Classification ============
    "bace": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether it can inhibit (True) the Beta-site Amyloid Precursor Protein Cleaving Enzyme 1 (BACE1) or cannot inhibit (False). Please answer with only True or False.
SMILES: {smiles}
BACE-1 Inhibit:""",

    "smol-property_prediction-bbbp": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether it can penetrate (True) the blood-brain barrier or not (False). Please answer with only True or False.
SMILES: {smiles}
BBB Penetration:""",

    "smol-property_prediction-clintox": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether the drug has been approved by FDA (True) or failed clinical trials for toxicity (False). Please answer with only True or False.
SMILES: {smiles}
FDA Approval:""",

    "smol-property_prediction-hiv": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether it can inhibit HIV replication (True) or not (False). Please answer with only True or False.
SMILES: {smiles}
HIV Inhibition:""",

    "smol-property_prediction-sider": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether it causes hepatobiliary disorders side effect (True) or not (False). Please answer with only True or False.
SMILES: {smiles}
Side Effect:""",

    "tox21": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether it shows toxicity (True) or not (False) in the Tox21 assay. Please answer with only True or False.
SMILES: {smiles}
Toxicity:""",

    "toxcast": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict whether it shows toxicity (True) or not (False) in the ToxCast assay. Please answer with only True or False.
SMILES: {smiles}
Toxicity:""",

    # ============ Regression - QM9 ============
    "qm9_homo": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the HOMO (Highest Occupied Molecular Orbital) energy in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
HOMO (Hartree):""",

    "qm9_lumo": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the LUMO (Lowest Unoccupied Molecular Orbital) energy in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
LUMO (Hartree):""",

    "qm9_homo_lumo_gap": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the HOMO-LUMO gap energy in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
Gap (Hartree):""",

    "qm9_dipole_moment": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the dipole moment in Debye units. Answer with only the numerical value.
SMILES: {smiles}
Dipole (Debye):""",

    "qm9_isotropic_polarizability": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the isotropic polarizability in Bohr^3 units. Answer with only the numerical value.
SMILES: {smiles}
Polarizability (Bohr^3):""",

    "qm9_electronic_spatial_extent": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the electronic spatial extent in Bohr^2 units. Answer with only the numerical value.
SMILES: {smiles}
Electronic Spatial Extent (Bohr^2):""",

    "qm9_zero_point_vibrational_energy": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the zero point vibrational energy in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
ZPVE (Hartree):""",

    "qm9_heat_capacity_298K": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the heat capacity at 298.15K in cal/(mol·K) units. Answer with only the numerical value.
SMILES: {smiles}
Cv (cal/mol·K):""",

    "qm9_internal_energy_298K": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the internal energy at 298.15K in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
U_298K (Hartree):""",

    "qm9_enthalpy_298K": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the enthalpy at 298.15K in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
H_298K (Hartree):""",

    "qm9_free_energy_298K": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the free energy at 298.15K in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
G_298K (Hartree):""",

    # ============ Regression - Alchemy ============
    "alchemy_homo": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the HOMO energy in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
HOMO (Hartree):""",

    "alchemy_lumo": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the LUMO energy in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
LUMO (Hartree):""",

    "alchemy_homo_lumo_gap": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the HOMO-LUMO gap energy in Hartree units. Answer with only the numerical value.
SMILES: {smiles}
Gap (Hartree):""",

    # ============ Regression - Solubility/Lipophilicity ============
    "smol-property_prediction-esol": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the aqueous solubility as log solubility (logS in mol/L). Answer with only the numerical value.
SMILES: {smiles}
LogS:""",

    "smol-property_prediction-lipo": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the lipophilicity as logD at pH 7.4. Answer with only the numerical value.
SMILES: {smiles}
LogD:""",

    "aqsol-logS": """You are an expert chemist, your task is to predict the property of molecules using your experienced chemical property prediction knowledge.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the aqueous solubility as logS in mol/L. Answer with only the numerical value.
SMILES: {smiles}
LogS:""",

    "pcqm_homo_lumo_gap": """You are an expert chemist, your task is to predict quantum mechanical properties of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the HOMO-LUMO gap energy in eV units. Answer with only the numerical value.
SMILES: {smiles}
Gap (eV):""",

    # ============ Reaction Prediction ============
    "forward_reaction_prediction": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the reactants SMILES, predict the major product of the reaction. Answer with only the product SMILES string.
Reactants: {smiles}
Product:""",

    "smol-forward_synthesis": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the reactants SMILES, predict the major product of the reaction. Answer with only the product SMILES string.
Reactants: {smiles}
Product:""",

    "retrosynthesis": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the product SMILES, predict the reactants needed to synthesize it. Answer with only the reactant SMILES (use '.' to separate multiple reactants).
Product: {smiles}
Reactants:""",

    "smol-retrosynthesis": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the product SMILES, predict the reactants needed to synthesize it. Answer with only the reactant SMILES (use '.' to separate multiple reactants).
Product: {smiles}
Reactants:""",

    "reagent_prediction": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the reactants and product of a reaction, predict the reagents or catalysts needed. Answer with only the reagent SMILES (use '.' to separate multiple reagents).
Reaction: {smiles}
Reagents:""",

    # Presto variants
    "presto-forward_reaction_prediction": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the reactants SMILES, predict the major product of the reaction. Answer with only the product SMILES string.
Reactants: {smiles}
Product:""",

    "presto-retrosynthesis": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the product SMILES, predict the reactants needed to synthesize it. Answer with only the reactant SMILES (use '.' to separate multiple reactants).
Product: {smiles}
Reactants:""",

    "presto-reagent_prediction": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the reactants and product of a reaction, predict the reagents or catalysts needed. Answer with only the reagent SMILES.
Reaction: {smiles}
Reagents:""",

    # Orderly variants
    "orderly-forward_reaction_prediction": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the reactants SMILES, predict the major product of the reaction. Answer with only the product SMILES string.
Reactants: {smiles}
Product:""",

    "orderly-retrosynthesis": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the product SMILES, predict the reactants needed to synthesize it. Answer with only the reactant SMILES (use '.' to separate multiple reactants).
Product: {smiles}
Reactants:""",

    "orderly-reagent_prediction": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Given the reactants and product of a reaction, predict the reagents or catalysts needed. Answer with only the reagent SMILES.
Reaction: {smiles}
Reagents:""",

    # ============ Molecule Generation (Text2Mol) ============
    "chebi-20-text2mol": """You are an expert chemist, your task is to generate molecules from descriptions.
Please strictly follow the format, no other information can be provided. Given the description of a molecule, generate the corresponding SMILES string. Answer with only the SMILES string.
Description: {description}
SMILES:""",

    "smol-molecule_generation": """You are an expert chemist, your task is to generate molecules from descriptions.
Please strictly follow the format, no other information can be provided. Given the description of a molecule, generate the corresponding SMILES string. Answer with only the SMILES string.
Description: {description}
SMILES:""",

    # ============ Molecule Captioning (Mol2Text) ============
    "chebi-20-mol2text": """You are an expert chemist, your task is to describe molecules.
Given the SMILES string of a molecule, provide a detailed description of its structure, functional groups, and properties.
SMILES: {smiles}
Description:""",

    "smol-molecule_captioning": """You are an expert chemist, your task is to describe molecules.
Given the SMILES string of a molecule, provide a detailed description of its structure, functional groups, and properties.
SMILES: {smiles}
Description:""",
}

# 기본 프롬프트 (매핑 안 된 task용)
DEFAULT_PROMPTS = {
    "classification": """You are an expert chemist, your task is to predict the property of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the property. Please answer with only True or False.
SMILES: {smiles}
Answer:""",

    "regression": """You are an expert chemist, your task is to predict the property of molecules.
Please strictly follow the format, no other information can be provided. Given the SMILES string of a molecule, predict the numerical property value. Answer with only the numerical value.
SMILES: {smiles}
Value:""",

    "reaction": """You are an expert chemist, your task is to predict chemical reactions.
Please strictly follow the format, no other information can be provided. Answer with only the SMILES string.
Input: {smiles}
Output:""",

    "text2mol": """You are an expert chemist, your task is to generate molecules from descriptions.
Please strictly follow the format, no other information can be provided. Answer with only the SMILES string.
Description: {description}
SMILES:""",

    "mol2text": """You are an expert chemist, your task is to describe molecules.
Given the SMILES string of a molecule, provide a description.
SMILES: {smiles}
Description:""",
}


# ============================================
# LlaSMol 스타일 프롬프트 템플릿 (SMolInstruct 원본 스타일)
# Uses <SMILES>...</SMILES> tags for canonicalization
# Reference: osunlp/SMolInstruct (https://huggingface.co/datasets/osunlp/SMolInstruct)
# LlaSMol GitHub: https://github.com/OSU-NLP-Group/LLM4Chem
#
# SMolInstruct uses natural question format, NOT instruction format.
# Response uses tags: <NUMBER>, <BOOLEAN>, <SMILES>, <MOLFORMULA>
# ============================================
LLASMOL_PROMPTS = {
    # ============ Classification ============
    # SMolInstruct style: natural question format
    "bace": """Can <SMILES> {smiles} </SMILES> inhibit human β-secretase 1 (BACE-1)?""",

    "smol-property_prediction-bbbp": """Is blood-brain barrier permeability (BBBP) a property of <SMILES> {smiles} </SMILES> ?""",

    "smol-property_prediction-clintox": """Is <SMILES> {smiles} </SMILES> toxic?""",

    "smol-property_prediction-hiv": """Can <SMILES> {smiles} </SMILES> serve as an inhibitor of HIV replication?""",

    "smol-property_prediction-sider": """Are there any known side effects of <SMILES> {smiles} </SMILES> affecting the heart?""",

    "tox21": """Is <SMILES> {smiles} </SMILES> toxic in the Tox21 assay?""",

    "toxcast": """Is <SMILES> {smiles} </SMILES> toxic in the ToxCast assay?""",

    # ============ Regression - Solubility/Lipophilicity ============
    # SMolInstruct example: "How soluble is <SMILES> CC(C)Cl </SMILES> ?"
    "smol-property_prediction-esol": """How soluble is <SMILES> {smiles} </SMILES> ?""",

    "smol-property_prediction-lipo": """Predict the octanol/water distribution coefficient logD under the circumstance of pH 7.4 for <SMILES> {smiles} </SMILES> .""",

    "aqsol-logS": """How soluble is <SMILES> {smiles} </SMILES> ?""",

    # ============ Regression - QM9 ============
    "qm9_homo": """What is the HOMO energy of <SMILES> {smiles} </SMILES> in Hartree?""",

    "qm9_lumo": """What is the LUMO energy of <SMILES> {smiles} </SMILES> in Hartree?""",

    "qm9_homo_lumo_gap": """What is the HOMO-LUMO gap of <SMILES> {smiles} </SMILES> in Hartree?""",

    "qm9_dipole_moment": """What is the dipole moment of <SMILES> {smiles} </SMILES> in Debye?""",

    "qm9_isotropic_polarizability": """What is the isotropic polarizability of <SMILES> {smiles} </SMILES> in Bohr³?""",

    "qm9_electronic_spatial_extent": """What is the electronic spatial extent of <SMILES> {smiles} </SMILES> in Bohr²?""",

    "qm9_zero_point_vibrational_energy": """What is the zero point vibrational energy of <SMILES> {smiles} </SMILES> in Hartree?""",

    "qm9_heat_capacity_298K": """What is the heat capacity at 298.15K of <SMILES> {smiles} </SMILES> in cal/(mol·K)?""",

    "qm9_internal_energy_298K": """What is the internal energy at 298.15K of <SMILES> {smiles} </SMILES> in Hartree?""",

    "qm9_enthalpy_298K": """What is the enthalpy at 298.15K of <SMILES> {smiles} </SMILES> in Hartree?""",

    "qm9_free_energy_298K": """What is the free energy at 298.15K of <SMILES> {smiles} </SMILES> in Hartree?""",

    # ============ Regression - Alchemy ============
    "alchemy_homo": """What is the HOMO energy of <SMILES> {smiles} </SMILES> in Hartree?""",

    "alchemy_lumo": """What is the LUMO energy of <SMILES> {smiles} </SMILES> in Hartree?""",

    "alchemy_homo_lumo_gap": """What is the HOMO-LUMO gap of <SMILES> {smiles} </SMILES> in Hartree?""",

    "pcqm_homo_lumo_gap": """What is the HOMO-LUMO gap of <SMILES> {smiles} </SMILES> in eV?""",

    # ============ Mol2Text ============
    # SMolInstruct example: "Describe this molecule: <SMILES> ... </SMILES>"
    "chebi-20-mol2text": """Describe this molecule: <SMILES> {smiles} </SMILES>""",

    "smol-molecule_captioning": """Describe this molecule: <SMILES> {smiles} </SMILES>""",

    # ============ Text2Mol ============
    # SMolInstruct example: "Give me a molecule that satisfies the conditions outlined in the description: ..."
    "chebi-20-text2mol": """Give me a molecule that satisfies the conditions outlined in the description: {description}""",

    "smol-molecule_generation": """Give me a molecule that satisfies the conditions outlined in the description: {description}""",

    # ============ Reaction - Forward Synthesis ============
    # SMolInstruct example: "<SMILES> ... </SMILES> Based on the reactants and reagents given above, suggest a possible product."
    "forward_reaction_prediction": """<SMILES> {smiles} </SMILES> Based on the reactants and reagents given above, suggest a possible product.""",

    "smol-forward_synthesis": """<SMILES> {smiles} </SMILES> Based on the reactants and reagents given above, suggest a possible product.""",

    "presto-forward_reaction_prediction": """<SMILES> {smiles} </SMILES> Based on the reactants and reagents given above, suggest a possible product.""",

    "orderly-forward_reaction_prediction": """<SMILES> {smiles} </SMILES> Based on the reactants and reagents given above, suggest a possible product.""",

    # ============ Reaction - Retrosynthesis ============
    # SMolInstruct example: "Identify possible reactants that could have been used to create the specified product. <SMILES> ... </SMILES>"
    "retrosynthesis": """Identify possible reactants that could have been used to create the specified product. <SMILES> {smiles} </SMILES>""",

    "smol-retrosynthesis": """Identify possible reactants that could have been used to create the specified product. <SMILES> {smiles} </SMILES>""",

    "presto-retrosynthesis": """Identify possible reactants that could have been used to create the specified product. <SMILES> {smiles} </SMILES>""",

    "orderly-retrosynthesis": """Identify possible reactants that could have been used to create the specified product. <SMILES> {smiles} </SMILES>""",

    # ============ Reaction - Reagent Prediction ============
    "reagent_prediction": """Given the reactants and product of a reaction <SMILES> {smiles} </SMILES>, predict the reagents or catalysts needed.""",

    "presto-reagent_prediction": """Given the reactants and product of a reaction <SMILES> {smiles} </SMILES>, predict the reagents or catalysts needed.""",

    "orderly-reagent_prediction": """Given the reactants and product of a reaction <SMILES> {smiles} </SMILES>, predict the reagents or catalysts needed.""",

 }


def selfies_to_smiles(selfies_str: str) -> Optional[str]:
    """SELFIES → SMILES (canonicalized)"""
    try:
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


def extract_user_prompt_from_llada(prompt_text: str) -> str:
    """
    LLaDA-8B 형식에서 user prompt 부분만 추출

    LLaDA format:
    <|startoftext|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    Returns:
        user_prompt 부분만 추출 (SELFIES, GRAPH 태그 포함)
    """
    # user<|end_header_id|>\n\n 와 <|eot_id|><|start_header_id|>assistant 사이의 내용 추출
    match = re.search(
        r'<\|start_header_id\|>user<\|end_header_id\|>\s*\n\n(.*?)<\|eot_id\|>',
        prompt_text,
        re.DOTALL
    )
    if match:
        return match.group(1).strip()

    # 매칭 실패시 원본 반환
    return prompt_text


def extract_user_prompt_from_author(prompt_text: str) -> str:
    """
    Author(원본) 데이터 형식에서 user prompt 부분만 추출

    Author format:
    <s>[INST] You are a helpful assistant...
    {user_prompt} [/INST]

    Returns:
        user_prompt 부분만 추출 (시스템 프롬프트 제외, SELFIES/GRAPH 태그 포함)
    """
    # <s>[INST] 제거
    prompt = re.sub(r'^<s>\s*\[INST\]\s*', '', prompt_text)
    # [/INST] 제거
    prompt = re.sub(r'\s*\[/INST\]\s*$', '', prompt)
    # 시스템 프롬프트 제거 (첫 번째 빈 줄 이후가 실제 user prompt)
    # "You are a helpful assistant..." 부분 제거
    prompt = re.sub(
        r'^You are a helpful assistant[^\n]*\n\n',
        '',
        prompt
    )
    return prompt.strip()


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
    else:
        return "unknown"


def format_prompt_for_galactica(prompt: str, selfies_str: str, task_name: str) -> str:
    """
    Format prompt for Galactica model (base model trained on scientific papers)
    원본 프롬프트를 유지하면서 Galactica 특화 변환만 적용:
    1. SELFIES → SMILES 변환
    2. [START_I_SMILES]...[END_I_SMILES] 포맷 적용

    Args:
        prompt: Original prompt text
        selfies_str: SELFIES representation of molecule
        task_name: Task name (e.g., 'smol-property_prediction-lipo')

    Returns:
        Formatted prompt for Galactica
    """
    # Convert SELFIES to SMILES (canonical)
    smiles_str = selfies_to_smiles(selfies_str)
    if smiles_str is None:
        try:
            smiles_str = sf.decoder(selfies_str)
        except:
            smiles_str = selfies_str

    # 원본 프롬프트 기반으로 Galactica 포맷 적용
    formatted = prompt

    # 1. <s>[INST]...[/INST] 포맷 제거 (Galactica는 instruction-tuned 모델이 아님)
    formatted = re.sub(r'^<s>\s*', '', formatted)
    formatted = re.sub(r'\[INST\]\s*', '', formatted)
    formatted = re.sub(r'\s*\[/INST\]\s*$', '', formatted)

    # 2. <GRAPH>...</GRAPH> 태그 제거 (Galactica는 그래프 이해 못함)
    formatted = re.sub(r'<GRAPH>.*?</GRAPH>', '', formatted)

    # 3. SELFIES 태그를 Galactica SMILES 포맷으로 변환
    # <SELFIES>...</SELFIES> → [START_I_SMILES]...[END_I_SMILES]
    # Note: SMILES에 \N 등 특수문자가 있을 수 있어 re.sub 대신 split/join 사용
    galactica_mol_format = f'[START_I_SMILES]{smiles_str}[END_I_SMILES]'

    if '<SELFIES>' in formatted and '</SELFIES>' in formatted:
        # re.sub 대신 split/join 사용하여 escape 문제 회피
        parts = re.split(r'<SELFIES>\s*[^<]*\s*</SELFIES>', formatted)
        formatted = galactica_mol_format.join(parts)
    elif selfies_str in formatted:
        formatted = formatted.replace(selfies_str, galactica_mol_format)

    # 4. 기존 SMILES 표기가 있으면 Galactica 포맷으로 변환
    smiles_match = re.search(r'SMILES:\s*([^\n\[]+?)(?=\n|$)', formatted)
    if smiles_match:
        formatted = formatted[:smiles_match.start()] + galactica_mol_format + formatted[smiles_match.end():]

    # 5. 공백 정리
    formatted = formatted.strip()

    # 6. Task별 Answer hint 추가 (Galactica가 더 잘 응답하도록)
    task_type = get_task_type(task_name)
    if task_type == "classification":
        # True/False로 답하도록 유도
        formatted += "\nAnswer (True or False):"
    elif task_type == "regression":
        formatted += "\nAnswer:"
    elif task_type == "reaction":
        # SMILES로 답하도록 명확히 지시
        formatted += "\nAnswer with SMILES only. Product: [START_I_SMILES]"
    elif task_type == "mol2text":
        formatted += "\nDescription:"
    elif task_type == "text2mol":
        # SMILES로 답하도록 명확히 지시
        formatted += "\nAnswer with SMILES only: [START_I_SMILES]"

    return formatted


def format_prompt_for_llama(prompt: str, selfies_str: str, task_name: str) -> str:
    """
    Format prompt for Llama models (instruction-tuned)
    Keeps instruction format but may adjust based on requirements
    """
    # For Llama, keep the original [INST]...[/INST] format
    # Just ensure SELFIES is present (no conversion to SMILES needed)
    return prompt


def format_prompt_for_mistral(prompt: str, selfies_str: str, task_name: str) -> str:
    """
    Format prompt for Mistral models (instruction-tuned)
    Similar to Llama format
    """
    return prompt


def format_prompt_for_gpt(prompt: str, selfies_str: str, task_name: str) -> str:
    """
    Format prompt for GPT models (API-based)
    May need specific formatting for API calls
    """
    # Remove instruction wrappers, keep clean format
    prompt = re.sub(r'<s>\[INST\]\s*', '', prompt)
    prompt = prompt.replace('[/INST]', '')
    return prompt.strip()


def format_prompt_for_llasmol(prompt: str, selfies_str: str, task_name: str) -> str:
    """
    Format prompt for LLaSMol model (Mistral-based instruction-tuned for chemistry)

    LlaSMolGeneration internally handles:
    - SMILES canonicalization via canonicalize_smiles_in_text()
    - [INST]...[/INST] wrapping via GeneralPrompter

    So we only need to:
    - Convert SELFIES to SMILES
    - Use <SMILES>...</SMILES> tags (LlaSMol format)
    - No [INST] wrapper needed

    Args:
        prompt: Original prompt text
        selfies_str: SELFIES representation of molecule
        task_name: Task name (e.g., 'smol-property_prediction-lipo')

    Returns:
        Formatted prompt for LLaSMol (without [INST] wrapper, with <SMILES> tags)
    """
    # Convert SELFIES to SMILES (canonical)
    smiles_str = selfies_to_smiles(selfies_str)
    if smiles_str is None:
        try:
            smiles_str = sf.decoder(selfies_str)
        except:
            smiles_str = selfies_str

    # Get task base name and type
    task_base = task_name.split("/")[0]
    task_type = get_task_type(task_name)

    # Extract description for text2mol tasks
    description = extract_description_from_prompt(prompt)

    # Get task-specific template (use LlaSMol-specific prompts with <SMILES> tags)
    template = LLASMOL_PROMPTS[task_base]

    # Format the template
    if "{description}" in template and description:
        formatted = template.format(description=description)
    elif "{smiles}" in template:
        formatted = template.format(smiles=smiles_str)
    else:
        formatted = template

    # Return without [INST] wrapper - LlaSMolGeneration adds it internally
    # Note: No Query/Response wrapper needed - SMolInstruct format is just the question itself
    return formatted


def format_prompt_for_chemdfm(prompt: str, selfies_str: str, task_name: str) -> str:
    """
    Format prompt for ChemDFM model (LLaMA-based instruction-tuned for chemistry)
    Uses ChemDFM official prompt style with task-specific templates.

    ChemDFM format: <s> [Round 0]\nHuman: ...\nAssistant:
    Reference: https://huggingface.co/OpenDFM/ChemDFM-v1.5-8B

    Args:
        prompt: Original prompt text
        selfies_str: SELFIES representation of molecule
        task_name: Task name (e.g., 'smol-property_prediction-lipo')

    Returns:
        Formatted prompt for ChemDFM
    """
    # Convert SELFIES to SMILES (canonical)
    smiles_str = selfies_to_smiles(selfies_str)
    if smiles_str is None:
        try:
            smiles_str = sf.decoder(selfies_str)
        except:
            smiles_str = selfies_str

    # Get task base name and type
    task_base = task_name.split("/")[0]
    task_type = get_task_type(task_name)

    # Extract description for text2mol tasks
    description = extract_description_from_prompt(prompt)

    # Get task-specific template
    if task_base in CHEMDFM_PROMPTS:
        template = CHEMDFM_PROMPTS[task_base]
    else:
        template = DEFAULT_PROMPTS.get(task_type, DEFAULT_PROMPTS["regression"])

    # Format the template
    if "{description}" in template and description:
        formatted = template.format(description=description)
    elif "{smiles}" in template:
        formatted = template.format(smiles=smiles_str)
    else:
        formatted = template

    # ChemDFM official wrapper format
    # NOTE: <s> is added by tokenizer, so we don't include it here
    return f"[Round 0]\nHuman: {formatted}\nAssistant:"


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


tasks = (
    CLASSIFICATION_BENCHMARKS
    + REGRESSION_BENCHMARKS
    + REACTION_BENCHMARKS
    + TEXT2MOL_BENCHMARKS
    + MOL2TEXT_BENCHMARKS
    + NAME_CONVERSION_BENCHMARKS
)

input_mol_string_pattern = re.compile("<SELFIES>.*?</SELFIES>")
graph_sequence = re.compile("<GRAPH>[<mol>]+?</GRAPH>")


def task2id(task):
    # task name to task id
    task2id = {k: i for i, k in enumerate(tasks)}
    return task2id[task]


def id2task(task_id):
    # task id to task name
    id2task = {i: k for i, k in enumerate(tasks)}
    return id2task[task_id]


class DataCollator(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer,
        padding=True,
        max_length=512,
        pad_to_multiple_of=None,
        return_tensors=None,
        train=True,
        args=None,
    ):
        super().__init__(
            tokenizer,
            padding=padding,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        self.train = train
        self.max_length = max_length
        self.tokenizer.padding_side = "left"
        self.mol_representation = args.mol_representation

        self.apply_molpo = args.train_molpo if self.train else args.eval_molpo

        self.projector_type = args.projector_type
        self.current_epoch = args.current_epoch
        self.args = args

        if self.mol_representation in ["string+graph", "graph_only"]:
            self.graph_collator = GraphCollater([], [])

    def select_mol_representation(self, prompt_text, mol_representation="string+graph"):
        if mol_representation == "string+graph":
            return prompt_text
        elif mol_representation == "string_only":
            string_only_prompt_text = [graph_sequence.sub("", p) for p in prompt_text]
            return string_only_prompt_text
        elif mol_representation == "graph_only":
            graph_only_prompt_text = [
                input_mol_string_pattern.sub("", p) for p in prompt_text
            ]
            return graph_only_prompt_text
        else:
            raise ValueError(
                "check /configs/*.yaml / mol_representation should be one of ['string+graph', 'string_only', 'graph_only']"
            )

    def enumerate_selfies(
        self,
        origin_selfies,
    ):
        origin_smiles = sf.decoder(origin_selfies)

        isomericSmiles = bool(self.args.isomericSmiles)
        canonical = bool(self.args.canonical)
        allHsExplicit = bool(self.args.allHsExplicit)

        processed_smiles = Chem.MolToSmiles(
            Chem.MolFromSmiles(origin_smiles),
            isomericSmiles=isomericSmiles,
            canonical=canonical,
            doRandom=not canonical,
            allHsExplicit=allHsExplicit,
            allBondsExplicit=False,
            kekuleSmiles=False,
        )
        processed_selfies = sf.encoder(processed_smiles)
        return processed_selfies

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # tasks = [task2id(sample["task"]) for sample in batch]  # task id
        temp = [sample for sample in batch]
        tasks = [task2id(sample["task"].split("/", 1)[0]) for sample in batch]
        task_names = [id2task(task) for task in tasks]
        raw_prompt_text = [sample["prompt_text"] for sample in batch]  # 원본 프롬프트 저장
        prompt_text = raw_prompt_text.copy()  # 변환용 복사본
        target_text = [sample["target_text"] for sample in batch]
        input_mol_strings = [sample["input_mol_string"] for sample in batch]
        list_selfies = [
            i.replace("<SELFIES> ", "").replace(" </SELFIES>", "")
            for i in input_mol_strings
        ]
        list_graphs = [
            Data(
                x=torch.tensor(sample["x"], dtype=torch.int64),
                edge_index=torch.tensor(sample["edge_index"], dtype=torch.int64),
                edge_attr=torch.tensor(sample["edge_attr"], dtype=torch.int64),
            )
            for sample in batch
        ]
        # for reagent prediction
        list_additional_graphs = [
            Data(
                x=torch.tensor(sample["additional_x"], dtype=torch.int64),
                edge_index=torch.tensor(
                    sample["additional_edge_index"], dtype=torch.int64
                ),
                edge_attr=torch.tensor(
                    sample["additional_edge_attr"], dtype=torch.int64
                ),
            )
            for sample in batch
        ]

        prompt_text = self.select_mol_representation(
            prompt_text, mol_representation=self.mol_representation
        )

        # Apply model-specific prompt formatting for benchmark mode
        if hasattr(self.args, 'benchmark') and self.args.benchmark:
            # Detect data format based on direct_data_root path
            data_root = getattr(self.args, 'direct_data_root', '') or ''
            is_llada_data = 'GSAI-ML-LLaDA-8B-Instruct' in data_root

            # Extract user prompt based on data format
            if is_llada_data:
                # LLaDA-8B format: extract user prompt from special tokens
                prompt_text = [extract_user_prompt_from_llada(p) for p in prompt_text]
            else:
                # Author (original) format: remove [INST] wrapper
                prompt_text = [
                    re.sub(r'<s>\[INST\]\s*', '', p).replace('[/INST]', '')
                    for p in prompt_text
                ]

            # Apply model-specific formatting
            # Use both llm_model and filename to identify the model
            model_name = self.args.llm_model.lower()
            filename = getattr(self.args, 'filename', '').lower()
            new_prompt_text = []

            # DEBUG: Print model detection info (first batch only)
            if not getattr(self, '_debug_printed', False):
                self._debug_printed = True

            for i, p in enumerate(prompt_text):
                selfies_str = list_selfies[i]
                task_name = task_names[i] if i < len(task_names) else ""

                # Route to appropriate formatter
                # Check filename first for LoRA-based models (e.g., LlaSMol uses base Mistral)
                if 'llasmol' in filename or 'llasmol' in model_name:
                    formatted_prompt = format_prompt_for_llasmol(p, selfies_str, task_name)
                elif 'galactica' in model_name or 'galactica' in filename:
                    formatted_prompt = format_prompt_for_galactica(p, selfies_str, task_name)
                elif 'chemdfm' in model_name or 'chemdfm' in filename:
                    formatted_prompt = format_prompt_for_chemdfm(p, selfies_str, task_name)
                elif 'llama' in model_name:
                    formatted_prompt = format_prompt_for_llama(p, selfies_str, task_name)
                elif 'mistral' in model_name:
                    formatted_prompt = format_prompt_for_mistral(p, selfies_str, task_name)
                elif 'gpt' in model_name:
                    formatted_prompt = format_prompt_for_gpt(p, selfies_str, task_name)
                else:
                    # Default: no special formatting
                    formatted_prompt = p

                new_prompt_text.append(formatted_prompt)

            prompt_text = new_prompt_text

        if not self.train and self.args.eval_modality_util in [
            "string",
            "graph",
        ]:
            shuffled_idx = []
            # shuffle the selfies_idx, guarantee that the selfies_idx is not in order
            for i in range(len(list_selfies)):
                idxs = np.random.choice(
                    range(len(list_selfies)), size=2, replace=False
                ).tolist()
                if i in idxs:
                    idxs.remove(i)
                shuffled_idx.append(idxs[0])

            if self.args.eval_modality_util == "string":
                processed_selfies = [list_selfies[i] for i in shuffled_idx]
                for i in range(len(prompt_text)):
                    assert (
                        list_selfies[i] in prompt_text[i]
                    ), f"{list_selfies[i]} not in {prompt_text[i]}"
                    prompt_text[i] = prompt_text[i].replace(
                        list_selfies[i], processed_selfies[i]
                    )

            if self.args.eval_modality_util == "graph":
                list_graphs = [list_graphs[i] for i in shuffled_idx]
                list_additional_graphs = [
                    list_additional_graphs[i] for i in shuffled_idx
                ]

        if self.args.selfies_enumeration:
            processed_selfies = [
                self.enumerate_selfies(list_selfies[i])
                for i in range(len(list_selfies))
            ]
            for i in range(len(prompt_text)):
                assert (
                    list_selfies[i] in prompt_text[i]
                ), f"{list_selfies[i]} not in {prompt_text[i]}"
                prompt_text[i] = prompt_text[i].replace(
                    list_selfies[i], processed_selfies[i]
                )
                list_selfies = processed_selfies

        if self.apply_molpo:
            if self.train:
                self.reject_cardinal = self.current_epoch
            else:
                self.reject_cardinal = 0

            prompt_text_reject = prompt_text.copy()

            if self.args.apply_preference_system_prompt:
                for i in range(len(prompt_text_reject)):
                    preference_system_prompt = "In the following problems, molecular graph is either accurate or inaccurate. Your predictions should be based primarily on careful understanding of the provided graph."
                    prompt_text_reject[i] = re.sub(
                        r"(?<=\[INST\]).*(?=\n\n)",
                        preference_system_prompt,
                        prompt_text_reject[i],
                    )

            prompt_text = prompt_text + prompt_text_reject * (
                self.args.molpo_batch_division - 1
            )
            if hasattr(self.args, "reject_label_mask") and self.args.reject_label_mask:
                reject_target_text = [sample[f"{self.reject_cardinal}-th_rejected_target_text"] for sample in batch]
                target_text = target_text + reject_target_text
            else:
                target_text = target_text * self.args.molpo_batch_division
            tasks = tasks * self.args.molpo_batch_division
            task_names = task_names * self.args.molpo_batch_division

            if "graph" in self.mol_representation:
                list_rejected_graphs = [
                    Data(
                        x=torch.tensor(
                            sample[f"{self.reject_cardinal}-th_rejected_x"],
                            dtype=torch.int64,
                        ),
                        edge_index=torch.tensor(
                            sample[f"{self.reject_cardinal}-th_rejected_edge_index"],
                            dtype=torch.int64,
                        ),
                        edge_attr=torch.tensor(
                            sample[f"{self.reject_cardinal}-th_rejected_edge_attr"],
                            dtype=torch.int64,
                        ),
                    )
                    for sample in batch
                ]
                # for reagent prediction
                list_rejected_additional_graphs = [
                    Data(
                        x=torch.tensor(
                            sample[f"{self.reject_cardinal}-th_additional_rejected_x"],
                            dtype=torch.int64,
                        ),
                        edge_index=torch.tensor(
                            sample[
                                f"{self.reject_cardinal}-th_additional_rejected_edge_index"
                            ],
                            dtype=torch.int64,
                        ),
                        edge_attr=torch.tensor(
                            sample[
                                f"{self.reject_cardinal}-th_additional_rejected_edge_attr"
                            ],
                            dtype=torch.int64,
                        ),
                    )
                    for sample in batch
                ]

                list_graphs = (
                    list_graphs * (self.args.molpo_batch_division - 1)
                    + list_rejected_graphs
                )
                list_additional_graphs = (
                    list_additional_graphs * (self.args.molpo_batch_division - 1)
                    + list_rejected_additional_graphs
                )

        # address <mol> token in prompt_text, for the case of using graph modality
        if self.projector_type == "mlp" and "graph" in self.mol_representation:
            for i in range(len(prompt_text)):
                if task_names[i] in ["reagent_prediction"]:
                    num_nodes_in_graph = list_graphs[i].x.size(0)
                    num_nodes_mol = "<mol>" * num_nodes_in_graph
                    mol_tokens_pattern = re.compile(r"(<mol>)+(?=</GRAPH>\|>>\|)")
                    assert mol_tokens_pattern.search(
                        prompt_text[i]
                    ), f"{prompt_text[i]}"
                    prompt_text[i] = mol_tokens_pattern.sub(
                        num_nodes_mol, prompt_text[i]
                    )

                    num_additional_nodes_in_graph = list_additional_graphs[i].x.size(0)
                    num_additional_nodes_mol = "<mol>" * num_additional_nodes_in_graph
                    additional_mol_tokens_pattern = re.compile(
                        r"(?<=\|>>\|<GRAPH>)(<mol>)+"
                    )
                    assert additional_mol_tokens_pattern.search(
                        prompt_text[i]
                    ), f"{prompt_text[i]}"
                    prompt_text[i] = additional_mol_tokens_pattern.sub(
                        num_additional_nodes_mol, prompt_text[i]
                    )
                elif task_names[i] in TEXT2MOL_BENCHMARKS:
                    # there is no input <mol> token
                    pass
                else:
                    num_nodes_in_graph = list_graphs[i].x.size(0)
                    num_nodes_mol = "<mol>" * num_nodes_in_graph
                    mol_tokens_pattern = re.compile("(<mol>)+")
                    assert mol_tokens_pattern.search(
                        prompt_text[i]
                    ), f"{prompt_text[i]}"
                    prompt_text[i] = mol_tokens_pattern.sub(
                        num_nodes_mol, prompt_text[i]
                    )

        self.tokenizer.padding_side = "left"
        prompt_tokenized = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        target_tokenized = self.tokenizer(
            target_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )

        full_input_ids = [
            p + t
            for p, t in zip(
                prompt_tokenized["input_ids"], target_tokenized["input_ids"]
            )
        ]
        full_attention_mask = [
            p + t
            for p, t in zip(
                prompt_tokenized["attention_mask"], target_tokenized["attention_mask"]
            )
        ]

        prompt_length = [len(p) for p in prompt_tokenized["input_ids"]]
        full_input_ids = [f_ids[: self.max_length] for f_ids in full_input_ids]
        full_attention_mask = [
            f_ids[: self.max_length] for f_ids in full_attention_mask
        ]

        self.tokenizer.padding_side = "left"
        features = self.tokenizer.pad(
            {"input_ids": full_input_ids, "attention_mask": full_attention_mask},
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        if not self.train:
            prompt_features = self.tokenizer.pad(
                {
                    "input_ids": [p for p in prompt_tokenized["input_ids"]],
                    "attention_mask": [p for p in prompt_tokenized["attention_mask"]],
                },
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )

            features["prompt_input_ids"] = prompt_features.input_ids  # ['input_ids']
            features["prompt_attention_mask"] = (
                prompt_features.attention_mask
            )  # ['attention_mask']

            self.tokenizer.padding_side = "right"
            gen_features = self.tokenizer.pad(
                {
                    "input_ids": [t for t in target_tokenized["input_ids"]],
                },
                padding=self.padding,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors=return_tensors,
            )
            gen_features.input_ids = gen_features.input_ids.masked_fill(
                gen_features.input_ids == self.tokenizer.pad_token_id, -100
            )
            features["gen_labels"] = gen_features.input_ids

            input_mol_strings_tokenized = self.tokenizer(
                input_mol_strings,
                truncation=False,
                max_length=self.max_length,
                padding=True,
                return_tensors=return_tensors,
                add_special_tokens=False,
            )

            features["input_mol_strings"] = input_mol_strings_tokenized.input_ids

        labels_ids = torch.full_like(features["input_ids"], self.tokenizer.pad_token_id)
        for i, target in enumerate(target_tokenized["input_ids"]):
            label = target
            if prompt_length[i] >= self.max_length:
                continue
            else:
                len_label = min(len(label), self.max_length - prompt_length[i])
                labels_ids[i, -len_label:] = torch.tensor(
                    label[:len_label], dtype=torch.int64
                )

        labels_ids = labels_ids.masked_fill(
            labels_ids == self.tokenizer.pad_token_id, -100
        )
        features["labels"] = labels_ids
        if self.apply_molpo:
            molpo_labels_ids = labels_ids.clone()
            for molpo_mask_id in self.tokenizer.molpo_mask_ids:
                molpo_labels_ids = molpo_labels_ids.masked_fill(
                    molpo_labels_ids == molpo_mask_id, -100
                )
            if hasattr(self.args, "reject_label_mask") and self.args.reject_label_mask:
                num_chosen = molpo_labels_ids.shape[0] // self.args.molpo_batch_division
                chosen_molpo_labels_ids = molpo_labels_ids.clone()[:num_chosen]
                reject_molpo_labels_ids = molpo_labels_ids.clone()[num_chosen:]

                chosen_molpo_labels_ids = chosen_molpo_labels_ids.masked_fill(
                    chosen_molpo_labels_ids == reject_molpo_labels_ids, -100
                )
                molpo_labels_ids = torch.cat(
                    (chosen_molpo_labels_ids, reject_molpo_labels_ids), dim=0
                )
            features["molpo_labels"] = molpo_labels_ids

        assert (
            features.input_ids.size(1) <= self.max_length
        ), f"features.input_ids.size(1)={features.input_ids.size(1)} > self.max_length={self.max_length}"
        assert (
            features.labels.size(1) <= self.max_length
        ), f"features.labels.size(1)={features.labels.size(1)} > self.max_length={self.max_length}"

        features["tasks"] = torch.tensor(tasks, dtype=torch.int16)
        # 문자열 리스트 저장 (transfer_batch_to_device에서 GPU 이동 스킵)
        features["raw_prompt_text"] = raw_prompt_text  # 원본 프롬프트 텍스트 (변환 전)
        features["prompt_text"] = prompt_text  # 변환된 프롬프트 텍스트
        features["target_text"] = target_text  # 타겟 텍스트
        if "graph" in self.mol_representation:
            graphs = self.graph_collator(list_graphs)
            additional_graphs = self.graph_collator(list_additional_graphs)
            features["graphs"] = graphs
            features["additional_graphs"] = additional_graphs
            features["is_mol_token"] = (
                features["input_ids"] == self.tokenizer.mol_token_id
            )
            if not self.train:
                features["prompt_is_mol_token"] = (
                    features["prompt_input_ids"].clone().detach() # .clone().detach() 수정됨
                    == self.tokenizer.mol_token_id
                )

        return features


def random_noise_selfies(selfies, tokenizer, sl_noise_ratio=0.3):
    selfies_ids = tokenizer.encode(selfies, add_special_tokens=False)
    total_selfies_token_ids = tokenizer.selfies_token_ids
    num_ids_to_replace = int(sl_noise_ratio * len(selfies_ids))
    replacing_random_ids = np.random.choice(
        total_selfies_token_ids, num_ids_to_replace, replace=True
    )

    # replace selfies_ids with randomly selected total_selfies_token_ids as many as num_ids_to_replace
    position_to_replace = np.random.choice(
        len(selfies_ids), num_ids_to_replace, replace=False
    )
    noised_selfies_ids = copy.deepcopy(selfies_ids)
    for i, replance_idx in enumerate(position_to_replace):
        noised_selfies_ids[replance_idx] = replacing_random_ids[i]

    noised_selfies = tokenizer.decode(
        noised_selfies_ids, skip_special_tokens=True
    ).replace(" ", "")
    return noised_selfies
