"""
3D-MoLM Generation Module
Wrapper for 3D-MoLM model inference compatible with Mol-LLM-Benchmark
"""

import os
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem import AllChem

from .config import UNIMOL_DEFAULTS, QFORMER_DEFAULTS, LORA_DEFAULTS

# 3D-MoLM uses 8 query tokens
NUM_MOL_TOKENS = 8


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class AttrDict(dict):
    """Dictionary that allows attribute-style access"""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value


def smiles2graph(smiles, dictionary):
    """Convert SMILES to 3D graph representation for UniMol"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if (np.asarray(atoms) == 'H').all():
        return None

    # Generate 3D coordinates
    res = AllChem.EmbedMolecule(mol)
    if res == 0:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass
        coordinates = mol.GetConformer().GetPositions()
    elif res == -1:
        mol_tmp = Chem.MolFromSmiles(smiles)
        AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000)
        mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(mol_tmp)
        except:
            pass
        try:
            coordinates = mol_tmp.GetConformer().GetPositions()
        except:
            # Fallback: use 2D coordinates if 3D fails
            AllChem.Compute2DCoords(mol)
            coordinates = mol.GetConformer().GetPositions()
            # Add zero z-coordinate
            coordinates = np.hstack([coordinates, np.zeros((len(coordinates), 1))])
    else:
        return None

    coordinates = coordinates.astype(np.float32)
    atoms = np.asarray(atoms)

    # Remove hydrogen atoms
    mask_hydrogen = atoms != "H"
    if sum(mask_hydrogen) > 0:
        atoms = atoms[mask_hydrogen]
        coordinates = coordinates[mask_hydrogen]

    # Atom vectors
    atom_vec = torch.from_numpy(dictionary.vec_index(atoms)).long()

    # Normalize coordinates
    coordinates = coordinates - coordinates.mean(axis=0)

    # Add special tokens (BOS/EOS)
    atom_vec = torch.cat([
        torch.LongTensor([dictionary.bos()]),
        atom_vec,
        torch.LongTensor([dictionary.eos()])
    ])
    coordinates = np.concatenate([
        np.zeros((1, 3)),
        coordinates,
        np.zeros((1, 3))
    ], axis=0)

    # Edge types
    edge_type = atom_vec.view(-1, 1) * len(dictionary) + atom_vec.view(1, -1)
    dist = distance_matrix(coordinates, coordinates).astype(np.float32)
    coordinates = torch.from_numpy(coordinates)
    dist = torch.from_numpy(dist)

    return atom_vec, dist, edge_type


class ThreeDMoLMGeneration:
    """
    3D-MoLM Generation class for benchmark testing.
    Loads the full 3D-MoLM model (UniMol + Q-Former + Llama2).
    """

    def __init__(
        self,
        ckpt_path,
        llm_model='meta-llama/Llama-2-7b-hf',
        bert_name='scibert',
        device=None,
    ):
        """
        Initialize 3D-MoLM model.

        Args:
            ckpt_path: Path to generalist.ckpt
            llm_model: Base LLM model path (Llama-2-7b-hf)
            bert_name: BERT model name for Q-Former (scibert)
            device: Target device
        """
        if device is None:
            device = get_device()
        self.device = device

        # Build args
        args = self._build_args(ckpt_path, llm_model, bert_name)

        # Import model components
        from .blip2_llama_inference import Blip2Llama

        # Load model
        print(f"[3D-MoLM] Loading model from {ckpt_path}")
        tensor_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = Blip2Llama(args).to(tensor_type)
        self.model.to(device)
        self.model.eval()

        self.tokenizer = self.model.llm_tokenizer
        self.dictionary = self.model.dictionary
        self.tensor_type = tensor_type

        print(f"[3D-MoLM] Model loaded successfully")

    def _build_args(self, ckpt_path, llm_model, bert_name):
        """Build arguments for model initialization"""
        args = AttrDict()

        # LLM settings
        args.llm_model = llm_model
        args.bert_name = bert_name
        args.lora_path = ckpt_path

        # UniMol settings
        for key, value in UNIMOL_DEFAULTS.items():
            args[key] = value

        # Q-Former settings
        for key, value in QFORMER_DEFAULTS.items():
            args[key] = value

        # LoRA settings
        for key, value in LORA_DEFAULTS.items():
            args[key] = value

        return args

    def _convert_mol_tokens(self, text, target_count=NUM_MOL_TOKENS):
        """
        Convert prompt with any number of <mol> tokens to exactly target_count tokens.
        3D-MoLM checkpoint uses 8 query tokens, so we need 8 <mol> tokens.
        Handles both "<mol><mol>" and "<mol> <mol>" patterns.
        """
        import re
        # Find all <mol> token sequences (with optional spaces between)
        # Pattern matches: <mol>, <mol> <mol>, <mol><mol>, etc.
        mol_pattern = r'(\s*<mol>\s*)+'

        def replace_mol_tokens(match):
            return ' ' + '<mol>' * target_count + ' '

        # Replace any sequence of <mol> tokens with exactly target_count
        converted = re.sub(mol_pattern, replace_mol_tokens, text)
        # Clean up extra spaces
        converted = re.sub(r'\s+', ' ', converted).strip()
        return converted

    def _tokenize(self, text):
        """Tokenize text input"""
        # Convert to 8 <mol> tokens if needed
        text = self._convert_mol_tokens(text)

        text_tokens = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=True
        )
        is_mol_token = text_tokens.input_ids == self.tokenizer.mol_token_id
        text_tokens['is_mol_token'] = is_mol_token

        # Debug: Always print mol token count for first few samples
        mol_count = is_mol_token.sum().item()
        print(f"[3D-MoLM DEBUG] mol_count={mol_count}, mol_token_id={self.tokenizer.mol_token_id}")
        if mol_count != NUM_MOL_TOKENS:
            print(f"[3D-MoLM] WARNING: Expected {NUM_MOL_TOKENS} <mol> tokens, got {mol_count}")
            print(f"[3D-MoLM] Converted text (last 200 chars): ...{text[-200:]}")

        return text_tokens

    def _prepare_graph(self, smiles):
        """Prepare 3D graph from SMILES"""
        result = smiles2graph(smiles, self.dictionary)
        if result is None:
            return None

        atom_vec, dist, edge_type = result
        atom_vec = atom_vec.unsqueeze(0).to(self.device)
        dist = dist.unsqueeze(0).to(self.tensor_type).to(self.device)
        edge_type = edge_type.unsqueeze(0).to(self.device)

        return (atom_vec, dist, edge_type)

    def generate(
        self,
        input_text,
        smiles_list=None,
        batch_size=1,
        max_new_tokens=128,
        min_new_tokens=32,
        num_beams=5,
        do_sample=False,
        repetition_penalty=1.2,
        length_penalty=1.0,
        **kwargs
    ):
        """
        Generate text outputs for given inputs.

        Args:
            input_text: List of prompt texts (should contain <mol> tokens)
            smiles_list: List of SMILES strings corresponding to each input
            batch_size: Batch size for generation
            max_new_tokens: Maximum new tokens to generate
            num_beams: Number of beams for beam search

        Returns:
            List of dicts with 'input_text', 'output' keys
        """
        if isinstance(input_text, str):
            input_text = [input_text]
        if smiles_list is None:
            smiles_list = [None] * len(input_text)
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]

        all_outputs = []

        for i in range(0, len(input_text), batch_size):
            batch_texts = input_text[i:i+batch_size]
            batch_smiles = smiles_list[i:i+batch_size]

            batch_outputs = []
            for text, smiles in zip(batch_texts, batch_smiles):
                try:
                    # Prepare graph
                    if smiles is not None:
                        graph = self._prepare_graph(smiles)
                    else:
                        graph = None

                    if graph is None:
                        batch_outputs.append({
                            'input_text': text,
                            'output': None,
                            'error': 'Failed to generate 3D graph'
                        })
                        continue

                    # Tokenize text
                    text_tokens = self._tokenize(text)
                    text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}

                    # Generate
                    with torch.no_grad():
                        output_text = self.model.generate(
                            graph,
                            text_tokens,
                            do_sample=do_sample,
                            num_beams=num_beams,
                            max_new_tokens=max_new_tokens,
                            min_new_tokens=min_new_tokens,
                            repetition_penalty=repetition_penalty,
                            length_penalty=length_penalty,
                        )

                    batch_outputs.append({
                        'input_text': text,
                        'output': output_text,
                    })

                except Exception as e:
                    batch_outputs.append({
                        'input_text': text,
                        'output': None,
                        'error': str(e)
                    })

            all_outputs.extend(batch_outputs)

        return all_outputs

    def generate_single(
        self,
        prompt,
        smiles,
        instruction=None,
        max_new_tokens=128,
        num_beams=5,
        **kwargs
    ):
        """
        Generate output for a single molecule.

        Args:
            prompt: Full prompt text with <mol> tokens, OR None to use default template
            smiles: SMILES string
            instruction: Instruction text (only used if prompt is None)

        Returns:
            Generated text string
        """
        if prompt is None:
            # Use default prompt template
            prompt_template = (
                "Below is an instruction that describes a task, paired with an input molecule. "
                "Write a response that appropriately completes the request.\n"
                "Instruction: {instruction}\n"
                "Input molecule: {smiles} <mol><mol><mol><mol><mol><mol><mol><mol>.\n"
                "Response: "
            )
            if instruction is None:
                instruction = "Describe this molecule."
            prompt = prompt_template.format(instruction=instruction, smiles=smiles)

        results = self.generate(
            [prompt],
            [smiles],
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            **kwargs
        )

        if results and results[0]['output']:
            return results[0]['output'][0] if isinstance(results[0]['output'], list) else results[0]['output']
        return None
