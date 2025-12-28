"""
Pure LLM Benchmark Model
Unified wrapper for various LLM models (Galactica, Llama, Mistral, etc.)
Used for zero-shot benchmark testing with text-only molecular representations (SMILES/SELFIES)

Also includes 3D-MoLM wrapper for 3D molecular representation benchmarking.
Also includes GPT-4/GPT-4o-mini wrapper for OpenAI API-based benchmarking.
"""

import logging
import os
import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    TaskType,
    PeftModel,
    get_peft_model,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    OPTForCausalLM,
    LlamaForCausalLM,
)

logger = logging.getLogger(__name__)


class LlaSMolBenchmarkLLM(nn.Module):
    """
    LlaSMol-specific benchmark wrapper using official LlaSMol generation code.
    Uses LlaSMolGeneration from LLM4Chem repository.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        peft_dir = getattr(args, 'peft_dir', '')
        llm_model = args.llm_model

        print(f"[LlaSMolBenchmarkLLM] Initializing LlaSMol model")
        print(f"[LlaSMolBenchmarkLLM] Base model: {llm_model}")
        print(f"[LlaSMolBenchmarkLLM] LoRA adapter: {peft_dir}")

        # Use official LlaSMol generation class
        from model.llasmol.generation import LlaSMolGeneration

        self.llasmol_generator = LlaSMolGeneration(
            model_name=peft_dir,  # e.g., 'osunlp/LlaSMol-Mistral-7B'
            base_model=llm_model,  # e.g., 'mistralai/Mistral-7B-v0.1'
        )

        # For compatibility with blip2_stage3
        self.llm_tokenizer = self.llasmol_generator.tokenizer
        self.llm_model = self.llasmol_generator.model

        print(f"[LlaSMolBenchmarkLLM] Model loaded and LoRA merged successfully")

    def init_tokenizer(self):
        """Return tokenizer for compatibility with blip2_stage3"""
        return self.llm_tokenizer

    def forward(self, batch):
        """
        Forward pass - text only (no graph embeddings)
        """
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        target_ids = batch.labels

        # Prepare targets to ignore pad tokens in loss calculation
        targets = target_ids.masked_fill(
            target_ids == self.llm_tokenizer.pad_token_id, -100
        )

        # Pure LLM forward pass
        outputs = self.llm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        # Calculate instance loss (per-sample loss)
        from model.blip2_stage3 import get_instance_loss
        loss_dict = get_instance_loss(logits=outputs.logits, labels=targets)

        results = {
            "loss": loss_dict["loss"],
            "instance_loss": loss_dict["instance_loss"],
            "logits": outputs.logits,
        }

        return results

    def generate(
        self,
        graphs=None,
        input_ids=None,
        attention_mask=None,
        is_mol_token=None,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_attentions=False,
        prompt_texts=None,  # Raw prompt texts for LlaSMol generation
        return_scores=False,  # Only True for classification tasks (ROC-AUC)
    ):
        """
        Generate text using official LlaSMol generation code.

        If prompt_texts is provided, use LlaSMolGeneration.generate() directly.
        Otherwise, fall back to standard generation with input_ids.
        """
        if prompt_texts is not None:
            # Use official LlaSMol generation
            generation_settings = {
                'num_beams': num_beams,
                'num_return_sequences': num_captions,
            }
            if do_sample:
                generation_settings['do_sample'] = True
                generation_settings['top_p'] = top_p
                generation_settings['temperature'] = temperature

            results = self.llasmol_generator.generate(
                input_text=prompt_texts,
                batch_size=len(prompt_texts),
                max_new_tokens=max_length,
                canonicalize_smiles=False,
                return_scores=return_scores,  # Only for classification tasks
                **generation_settings,
            )

            # Extract predictions and all_logits
            predictions = []
            all_logits_list = []
            for result in results:
                if result['output'] is not None:
                    predictions.append(result['output'][0] if result['output'] else "")
                else:
                    predictions.append("")
                # Collect all_logits (seq_len, vocab) for each sample
                all_logits_list.append(result.get('all_logits'))

            # Create output object compatible with blip2_stage3
            class GenerateOutput:
                pass
            outputs = GenerateOutput()
            outputs.predictions = predictions
            outputs.sequences = None
            outputs.attentions = None

            # Construct logits tensor from all_logits: (batch, seq_len, vocab_size)
            # For classification ROC-AUC, convert_logit2binary_prob finds <BOOLEAN> position
            valid_logits = [l for l in all_logits_list if l is not None]
            if valid_logits:
                # Pad sequences to same length and stack
                max_seq_len = max(l.shape[0] for l in valid_logits)
                vocab_size = valid_logits[0].shape[1]
                device = valid_logits[0].device

                padded_logits = []
                for l in valid_logits:
                    if l.shape[0] < max_seq_len:
                        # Pad with zeros
                        padding = torch.zeros(max_seq_len - l.shape[0], vocab_size, device=device, dtype=l.dtype)
                        l = torch.cat([l, padding], dim=0)
                    padded_logits.append(l)

                outputs.logits = torch.stack(padded_logits, dim=0)  # (batch, seq_len, vocab_size)
            else:
                outputs.logits = None

            return outputs

        # Fallback: standard generation with input_ids
        outputs = self.llm_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_new_tokens=min_length,
            eos_token_id=self.llm_tokenizer.eos_token_id,
            pad_token_id=self.llm_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            output_scores=True,
            return_dict_in_generate=True,
            output_attentions=output_attentions,
        )

        # Process logits
        batch_size, sequence_length = outputs.sequences.shape
        logits_stacked = torch.zeros(
            batch_size,
            0,
            self.llm_model.config.vocab_size,
            device=outputs.sequences.device,
        )

        if hasattr(outputs, 'scores') and outputs.scores:
            for i in range(len(outputs.scores)):
                logits = outputs.scores[i].unsqueeze(1)
                if num_beams > 1:
                    logits = logits.view(batch_size, num_beams, -1).max(dim=1).values.unsqueeze(1)
                logits_stacked = torch.cat([logits_stacked, logits], dim=1)

        outputs.logits = logits_stacked

        # Decode predictions
        prompt_length = input_ids.shape[1]
        generated_sequences = outputs.sequences[:, prompt_length:]

        output_text = self.llm_tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=False
        )
        output_text = [text.strip() for text in output_text]
        outputs.predictions = output_text

        return outputs


class BenchmarkLLM(nn.Module):
    """
    Pure LLM wrapper for benchmark testing
    Supports: Galactica, Llama, Mistral, and other HuggingFace causal LMs
    Only uses text-based molecular representations (SMILES/SELFIES strings)
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        llm_model = args.llm_model
        tune_llm = args.tune_llm
        peft_dir = getattr(args, 'peft_dir', '')

        print(f"[BenchmarkLLM] Initializing pure LLM: {llm_model}")
        print(f"[BenchmarkLLM] Mode: {tune_llm}")
        print(f"[BenchmarkLLM] Mol representation: {args.mol_representation}")

        # Initialize tokenizer
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model,
            use_fast=False,
            padding_side="left",
        )
        self.add_necessary_tokens()

        # Load LLM model based on model type
        self.llm_model = self.load_llm_model(llm_model)

        # Resize token embeddings for added tokens
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        # Configure LoRA if needed
        self.tune_llm = tune_llm
        if tune_llm == "lora":
            if peft_dir:
                self.llm_model = PeftModel.from_pretrained(
                    self.llm_model, peft_dir, is_trainable=True
                )
                print(f"[BenchmarkLLM] Loaded LoRA weights from: {peft_dir}")
            else:
                if hasattr(args, 'peft_config') and args.peft_config:
                    peft_config = LoraConfig.from_json_file(args.peft_config)
                else:
                    peft_config = LoraConfig(
                        target_modules=self.get_lora_target_modules(llm_model),
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                    )
                self.llm_model = get_peft_model(self.llm_model, peft_config)
                self.llm_model.print_trainable_parameters()
        elif tune_llm == "freeze":
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False
            print("[BenchmarkLLM] All LLM parameters frozen")
        elif tune_llm == "full":
            print("[BenchmarkLLM] Full fine-tuning mode")
        else:
            raise NotImplementedError(f"tune_llm={tune_llm} not supported")

    def load_llm_model(self, llm_model):
        """Load LLM model based on model type"""
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if "galactica" in llm_model.lower():
            # Galactica uses OPT architecture
            model = OPTForCausalLM.from_pretrained(
                llm_model,
                torch_dtype=dtype,
            )
            print(f"[BenchmarkLLM] Loaded Galactica (OPT) model")
        elif "llama" in llm_model.lower():
            # Llama models
            model = LlamaForCausalLM.from_pretrained(
                llm_model,
                torch_dtype=dtype,
            )
            print(f"[BenchmarkLLM] Loaded Llama model")
        elif "mistral" in llm_model.lower():
            # Mistral uses auto loading
            model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                torch_dtype=dtype,
            )
            print(f"[BenchmarkLLM] Loaded Mistral model")
        else:
            # Generic auto loading for other models
            model = AutoModelForCausalLM.from_pretrained(
                llm_model,
                torch_dtype=dtype,
            )
            print(f"[BenchmarkLLM] Loaded model using AutoModelForCausalLM")

        return model

    def get_lora_target_modules(self, llm_model):
        """Get LoRA target modules based on model architecture"""
        if "galactica" in llm_model.lower():
            # OPT architecture
            return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
        elif "llama" in llm_model.lower():
            # Llama architecture
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "mistral" in llm_model.lower():
            # Mistral architecture (similar to Llama)
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            # Default: common attention layers
            return ["q_proj", "k_proj", "v_proj", "o_proj"]

    def add_necessary_tokens(self):
        """Add special tokens for the model and SELFIES"""
        # Pad token
        if not self.llm_tokenizer.pad_token:
            self.llm_tokenizer.add_special_tokens({"pad_token": "<pad>"})

        # EOS token (Galactica uses newline)
        if not self.llm_tokenizer.eos_token:
            if "galactica" in self.args.llm_model.lower():
                self.llm_tokenizer.add_special_tokens({"eos_token": "\n"})
            else:
                # Most models already have eos_token, but set to </s> if missing
                self.llm_tokenizer.add_special_tokens({"eos_token": "</s>"})

        # Add SELFIES tokens if needed
        if self.args.add_selfies_tokens:
            with open(self.args.selfies_token_path, "r") as f:
                selfies_tokens = f.readlines()
                selfies_tokens = [token.strip() for token in selfies_tokens]
            self.llm_tokenizer.add_tokens(selfies_tokens)
            self.llm_tokenizer.selfies_token_ids = [
                self.llm_tokenizer.convert_tokens_to_ids(token)
                for token in selfies_tokens
            ]
            self.llm_tokenizer.added_selfies_tokens = selfies_tokens
            print(f"[BenchmarkLLM] Added {len(selfies_tokens)} SELFIES tokens")

    def init_tokenizer(self):
        """Return tokenizer for compatibility with blip2_stage3"""
        return self.llm_tokenizer

    def forward(self, batch):
        """
        Forward pass - text only (no graph embeddings)

        Args:
            batch: DataLoader batch containing:
                - input_ids: tokenized input sequences
                - attention_mask: attention mask
                - labels: target token ids

        Returns:
            dict with keys: loss, instance_loss, logits
        """
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        target_ids = batch.labels

        # Prepare targets to ignore pad tokens in loss calculation
        targets = target_ids.masked_fill(
            target_ids == self.llm_tokenizer.pad_token_id, -100
        )

        # Pure LLM forward pass (no graph embeddings)
        outputs = self.llm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )

        # Calculate instance loss (per-sample loss)
        from model.blip2_stage3 import get_instance_loss
        loss_dict = get_instance_loss(logits=outputs.logits, labels=targets)

        results = {
            "loss": loss_dict["loss"],
            "instance_loss": loss_dict["instance_loss"],
            "logits": outputs.logits,
        }

        return results

    def generate(
        self,
        graphs=None,  # Not used, kept for API compatibility
        input_ids=None,
        attention_mask=None,
        is_mol_token=None,  # Not used, kept for API compatibility
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_attentions=False,
        prompt_texts=None,  # Not used, kept for API compatibility with LlaSMol
        return_scores=False,  # Not used, kept for API compatibility with LlaSMol
    ):
        """
        Generate text - pure LLM generation (no graph embeddings)

        Args:
            input_ids: tokenized prompt
            attention_mask: attention mask
            num_beams: beam search beam count
            max_length: max generation length
            ... (other generation parameters)

        Returns:
            generation outputs with predictions
        """
        # Pure LLM generation (no graph embeddings injection)
        # output_scores: True for classification tasks (ROC-AUC), False otherwise for efficiency
        outputs = self.llm_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_new_tokens=min_length,
            eos_token_id=self.llm_tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            output_scores=return_scores,
            return_dict_in_generate=True,
            output_attentions=output_attentions,
        )

        # Process logits
        batch_size, sequence_length = outputs.sequences.shape
        logits_stacked = torch.zeros(
            batch_size,
            0,
            self.llm_model.config.vocab_size,
            device=outputs.sequences.device,
        )

        if hasattr(outputs, 'scores') and outputs.scores:
            for i in range(len(outputs.scores)):
                logits = outputs.scores[i].unsqueeze(1)
                if num_beams > 1:
                    logits = logits.view(batch_size, num_beams, -1).max(dim=1).values.unsqueeze(1)
                logits_stacked = torch.cat([logits_stacked, logits], dim=1)

        outputs.logits = logits_stacked

        # Decode predictions (only generated tokens, excluding input prompt)
        # outputs.sequences includes both input and generated tokens
        # We need to slice off the input_ids portion
        prompt_length = input_ids.shape[1]
        generated_sequences = outputs.sequences[:, prompt_length:]

        output_text = self.llm_tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=False
        )
        output_text = [text.strip() for text in output_text]
        outputs.predictions = output_text

        return outputs


class MoLM3DBenchmark(nn.Module):
    """
    3D-MoLM Benchmark wrapper.
    Uses 3D molecular representation (UniMol) + Q-Former + Llama-2-7B with LoRA.

    Directly uses Blip2Llama like test_inference.ipynb for correct inference.

    Reference: "Towards 3D Molecule-Text Interpretation in Language Models" (ICLR 2024)
    GitHub: https://github.com/lsh0520/3D-MoLM
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        ckpt_path = getattr(args, 'molm_3d_ckpt', None)
        llm_model = getattr(args, 'llm_model', 'meta-llama/Llama-2-7b-hf')
        bert_name = getattr(args, 'bert_name', 'scibert')

        if ckpt_path is None:
            raise ValueError("molm_3d_ckpt must be specified for 3D-MoLM model")

        print(f"[MoLM3DBenchmark] Initializing 3D-MoLM model (direct Blip2Llama)")
        print(f"[MoLM3DBenchmark] Checkpoint: {ckpt_path}")
        print(f"[MoLM3DBenchmark] Base LLM: {llm_model}")

        # Build args like test_inference.ipynb
        from model.molm_3d.blip2_llama_inference import Blip2Llama
        from model.molm_3d.config import UNIMOL_DEFAULTS, QFORMER_DEFAULTS, LORA_DEFAULTS

        # Create args object
        class AttrDict(dict):
            def __getattr__(self, key):
                try:
                    return self[key]
                except KeyError:
                    raise AttributeError(key)
            def __setattr__(self, key, value):
                self[key] = value

        model_args = AttrDict()
        # Use local paths like test_inference.ipynb for compatibility
        import os
        molm_3d_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'molm_3d')
        model_args.llm_model = os.path.join(molm_3d_dir, 'all_checkpoints/llama-2-7b-hf')
        model_args.bert_name = os.path.join(molm_3d_dir, 'all_checkpoints/scibert_scivocab_uncased')
        model_args.lora_path = ckpt_path
        print(f"[MoLM3DBenchmark] Using local llm_model: {model_args.llm_model}")
        print(f"[MoLM3DBenchmark] Using local bert_name: {model_args.bert_name}")

        # UniMol settings
        for key, value in UNIMOL_DEFAULTS.items():
            model_args[key] = value
        # Q-Former settings
        for key, value in QFORMER_DEFAULTS.items():
            model_args[key] = value
        # LoRA settings
        for key, value in LORA_DEFAULTS.items():
            model_args[key] = value

        # Load model like test_inference.ipynb
        # Note: Keep model in bf16 like test_inference.ipynb, but avoid Lightning's mixed precision
        self.tensor_type = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Blip2Llama(model_args).to(self.tensor_type)
        self.model.to(self.device)
        self.model.eval()

        print(f"[MoLM3DBenchmark] Model dtype: {self.tensor_type}")

        # For compatibility with blip2_stage3
        self.llm_tokenizer = self.model.llm_tokenizer
        self.llm_model = self.model.llm_model
        self.dictionary = self.model.dictionary

        print(f"[MoLM3DBenchmark] Model loaded successfully")

    def init_tokenizer(self):
        """Return tokenizer for compatibility with blip2_stage3"""
        return self.llm_tokenizer

    def _tokenize(self, text):
        """Tokenize text input (like test_inference.ipynb)"""
        import re
        # Convert any number of <mol> tokens to exactly 8 (like test_inference.ipynb)
        mol_pattern = r'(\s*<mol>\s*)+'
        text = re.sub(mol_pattern, ' ' + '<mol>' * 8 + ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        text_tokens = self.llm_tokenizer(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            return_attention_mask=True,
            return_token_type_ids=True
        )
        is_mol_token = text_tokens.input_ids == self.llm_tokenizer.mol_token_id
        text_tokens['is_mol_token'] = is_mol_token

        # Verify mol token count
        mol_count = is_mol_token.sum().item()
        if mol_count != 8:
            print(f"[MoLM3DBenchmark] WARNING: Expected 8 <mol> tokens, got {mol_count}")

        return text_tokens

    def forward(self, batch):
        """
        Forward pass for 3D-MoLM - computes loss and logits.
        Processes each sample individually and aggregates results.
        """
        # Support both dict and object-style access
        if hasattr(batch, 'input_ids'):
            input_ids = batch.input_ids
            attention_mask = batch.attention_mask
            labels = batch.labels
            molm_3d_graphs = getattr(batch, 'molm_3d_graphs', None) or batch.get("molm_3d_graphs", None) if hasattr(batch, 'get') else None
        else:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            molm_3d_graphs = batch.get("molm_3d_graphs", None)

        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Get pre-computed 3D graphs from batch (prepared by DataCollator)
        if molm_3d_graphs is None:
            molm_3d_graphs = [None] * batch_size

        # Process each sample and collect results
        all_losses = []
        all_logits = []
        from model.blip2_stage3 import get_instance_loss

        for idx in range(batch_size):
            graph_tuple = molm_3d_graphs[idx] if idx < len(molm_3d_graphs) else None

            if graph_tuple is None:
                # No valid graph - use zero loss
                all_losses.append(torch.tensor(0.0, device=device))
                all_logits.append(None)
                continue

            # Move graph to device
            # Note: dist must stay float32 (not bfloat16) for UniMol encoder
            atom_vec, dist, edge_type = graph_tuple
            graph = (
                atom_vec.unsqueeze(0).to(device),
                dist.unsqueeze(0).to(device),  # Keep float32 - don't convert dtype
                edge_type.unsqueeze(0).to(device),
            )

            # Get single sample data
            sample_input_ids = input_ids[idx:idx+1]
            sample_attention_mask = attention_mask[idx:idx+1]
            sample_labels = labels[idx:idx+1]
            is_mol_token = (sample_input_ids == self.llm_tokenizer.mol_token_id)

            text_batch = {
                'input_ids': sample_input_ids,
                'attention_mask': sample_attention_mask,
                'is_mol_token': is_mol_token,
            }

            # Forward through Blip2Llama (direct call like test_inference.ipynb)
            outputs = self.model.forward(
                graph_batch=graph,
                text_batch=text_batch,
                labels=sample_labels,
            )

            all_losses.append(outputs.get('loss', torch.tensor(0.0, device=device)))
            all_logits.append(outputs.get('logits'))

        # Aggregate losses
        valid_losses = [l for l in all_losses if l.item() > 0]
        if valid_losses:
            total_loss = torch.stack(valid_losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=device)

        # Calculate instance loss from labels directly
        instance_loss = torch.zeros(batch_size, device=device)
        for idx in range(batch_size):
            if all_logits[idx] is not None:
                sample_labels = labels[idx:idx+1]
                loss_dict = get_instance_loss(logits=all_logits[idx], labels=sample_labels)
                instance_loss[idx] = loss_dict["instance_loss"][0]

        # For logits, we need to pad to same sequence length if returning
        # For now, return None for logits as it's complex to batch variable-length sequences
        return {
            "loss": total_loss,
            "instance_loss": instance_loss,
            "logits": None,  # Cannot easily batch variable-length logits
        }

    def _pad_graphs(self, graph_tuples):
        """Pad graphs to same size for batching.

        Args:
            graph_tuples: List of (atom_vec, dist, edge_type) tuples

        Returns:
            Batched graph tuple (atom_vec_batch, dist_batch, edge_type_batch)
        """
        # Find max atom count
        max_atoms = max(g[0].shape[0] for g in graph_tuples)

        atom_vecs = []
        dists = []
        edge_types = []

        for atom_vec, dist, edge_type in graph_tuples:
            num_atoms = atom_vec.shape[0]
            pad_size = max_atoms - num_atoms

            if pad_size > 0:
                # Pad atom_vec with padding token (0)
                atom_vec = torch.nn.functional.pad(atom_vec, (0, pad_size), value=0)
                # Pad dist with large values (so attention ignores padded atoms)
                dist = torch.nn.functional.pad(dist, (0, pad_size, 0, pad_size), value=1e6)
                # Pad edge_type with 0
                edge_type = torch.nn.functional.pad(edge_type, (0, pad_size, 0, pad_size), value=0)

            atom_vecs.append(atom_vec)
            dists.append(dist)
            edge_types.append(edge_type)

        return (
            torch.stack(atom_vecs),
            torch.stack(dists),
            torch.stack(edge_types),
        )

    def _tokenize_batch(self, texts):
        """Tokenize a batch of texts with <mol> token handling."""
        import re

        processed_texts = []
        for text in texts:
            # Convert any number of <mol> tokens to exactly 8
            mol_pattern = r'(\s*<mol>\s*)+'
            text = re.sub(mol_pattern, ' ' + '<mol>' * 8 + ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            processed_texts.append(text)

        text_tokens = self.llm_tokenizer(
            processed_texts,
            add_special_tokens=True,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024,
            return_attention_mask=True,
            return_token_type_ids=True
        )
        is_mol_token = text_tokens.input_ids == self.llm_tokenizer.mol_token_id
        text_tokens['is_mol_token'] = is_mol_token

        return text_tokens

    def generate(
        self,
        graphs=None,
        input_ids=None,
        attention_mask=None,
        is_mol_token=None,
        do_sample=False,
        num_beams=1,  # Use 1 beam like test_inference.ipynb
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,  # Use 1.0 like test_inference.ipynb
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        output_attentions=False,
        prompt_texts=None,  # Raw prompt texts with <mol> tokens
        molm_3d_graphs=None,  # Pre-computed 3D graphs from DataCollator
        return_scores=False,
    ):
        """
        Generate text using 3D-MoLM with batch processing.

        Args:
            prompt_texts: List of prompt texts containing <mol> tokens
            molm_3d_graphs: List of pre-computed 3D graphs (atom_vec, dist, edge_type)
            num_beams: Number of beams for beam search
            max_length: Maximum new tokens to generate

        Returns:
            Generation outputs with predictions
        """
        if prompt_texts is None:
            raise ValueError("prompt_texts must be provided for 3D-MoLM generation")

        if molm_3d_graphs is None:
            molm_3d_graphs = [None] * len(prompt_texts)

        # Separate valid and invalid samples
        valid_indices = []
        valid_prompts = []
        valid_graphs = []

        for i, (prompt, graph_tuple) in enumerate(zip(prompt_texts, molm_3d_graphs)):
            if graph_tuple is not None:
                valid_indices.append(i)
                valid_prompts.append(prompt)
                valid_graphs.append(graph_tuple)

        # Initialize predictions with empty strings
        predictions = [""] * len(prompt_texts)

        if not valid_graphs:
            # No valid samples
            class GenerateOutput:
                pass
            outputs = GenerateOutput()
            outputs.predictions = predictions
            outputs.sequences = None
            outputs.attentions = None
            outputs.logits = None
            return outputs

        # Single-sample processing (batch processing disabled due to graph padding issues)
        all_logits = []
        for i, idx in enumerate(valid_indices):
            try:
                graph_tuple = valid_graphs[i]
                prompt = valid_prompts[i]

                atom_vec, dist, edge_type = graph_tuple
                graph = (
                    atom_vec.unsqueeze(0).to(self.device),
                    dist.unsqueeze(0).to(self.device),
                    edge_type.unsqueeze(0).to(self.device),
                )

                text_tokens = self._tokenize(prompt)
                text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}

                with torch.no_grad():
                    output_text, sample_logits = self.model.generate(
                        graph,
                        text_tokens,
                        max_new_tokens=max_length,
                        min_new_tokens=min_length,
                        num_beams=num_beams,
                        do_sample=do_sample,
                        repetition_penalty=repetition_penalty,
                    )

                if isinstance(output_text, list) and output_text:
                    predictions[idx] = output_text[0]
                else:
                    predictions[idx] = output_text if output_text else ""

                # Collect logits for this sample
                all_logits.append((idx, sample_logits))

            except Exception as e:
                print(f"[MoLM3DBenchmark] Sample {idx} error: {e}")
                predictions[idx] = ""
                all_logits.append((idx, None))

        # Create output object compatible with blip2_stage3
        class GenerateOutput:
            pass
        outputs = GenerateOutput()
        outputs.predictions = predictions
        outputs.sequences = None
        outputs.attentions = None

        # Pad and stack logits if available
        valid_logits = [(idx, l) for idx, l in all_logits if l is not None]
        if valid_logits:
            # Find max length for padding
            max_len = max(l.shape[1] for _, l in valid_logits)
            vocab_size = valid_logits[0][1].shape[2]

            # Create full logits tensor for all samples
            full_logits = torch.zeros(len(prompt_texts), max_len, vocab_size, device=self.device)
            for idx, l in valid_logits:
                seq_len = l.shape[1]
                full_logits[idx, :seq_len, :] = l.squeeze(0)

            outputs.logits = full_logits
        else:
            outputs.logits = None

        return outputs


class GPT4BenchmarkLLM(nn.Module):
    """
    OpenAI GPT-4/GPT-4o-mini Benchmark wrapper.
    Uses OpenAI API for inference - no local GPU required.

    Supported models:
    - gpt-4-0613: Original GPT-4 (2023.06)
    - gpt-4-turbo: GPT-4 Turbo (cheaper, faster)
    - gpt-4o: GPT-4o (multimodal, cheapest GPT-4)
    - gpt-4o-mini: GPT-4o-mini (very cheap, good performance)
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Get model name from args
        self.model_name = getattr(args, 'llm_model', 'gpt-4o-mini')
        self.api_key = os.environ.get('OPENAI_API_KEY', None)

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Please set it with: export OPENAI_API_KEY='sk-...'"
            )

        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "openai package not installed. "
                "Please install with: pip install openai"
            )

        # Generation parameters
        self.temperature = getattr(args, 'temperature', 0.0)  # Deterministic by default
        self.max_tokens = getattr(args, 'gen_max_len', 256)
        self.top_p = getattr(args, 'top_p', 1.0)

        # Rate limiting
        self.request_delay = getattr(args, 'request_delay', 0.0)  # seconds between requests

        # Create a dummy tokenizer for compatibility
        self.llm_tokenizer = self._create_dummy_tokenizer()
        self.llm_model = None  # No local model

        print(f"[GPT4BenchmarkLLM] Initialized with model: {self.model_name}")
        print(f"[GPT4BenchmarkLLM] Temperature: {self.temperature}")
        print(f"[GPT4BenchmarkLLM] Max tokens: {self.max_tokens}")

    def _create_dummy_tokenizer(self):
        """Create a dummy tokenizer object for compatibility with blip2_stage3."""
        class DummyTokenizer:
            def __init__(self):
                self.pad_token = "<pad>"
                self.pad_token_id = 0
                self.eos_token = "</s>"
                self.eos_token_id = 1
                self.bos_token = "<s>"
                self.bos_token_id = 2

            def __call__(self, texts, **kwargs):
                """Dummy tokenization - just return placeholder tensors."""
                if isinstance(texts, str):
                    texts = [texts]
                batch_size = len(texts)
                # Return dummy tensors
                return {
                    'input_ids': torch.zeros(batch_size, 1, dtype=torch.long),
                    'attention_mask': torch.ones(batch_size, 1, dtype=torch.long),
                }

            def batch_decode(self, token_ids, **kwargs):
                """Dummy decode - not used for GPT-4."""
                return [""] * len(token_ids)

            def decode(self, token_ids, **kwargs):
                """Dummy decode - not used for GPT-4."""
                return ""

        return DummyTokenizer()

    def init_tokenizer(self):
        """Return tokenizer for compatibility with blip2_stage3."""
        return self.llm_tokenizer

    def forward(self, batch):
        """
        Forward pass - not supported for API-based models.
        GPT-4 is inference-only via API.
        """
        raise NotImplementedError(
            "GPT-4 does not support forward pass. Use generate() for inference."
        )

    def _call_api(self, prompt: str) -> str:
        """Call OpenAI API with a single prompt."""
        import time

        if self.request_delay > 0:
            time.sleep(self.request_delay)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"[GPT4BenchmarkLLM] API call failed: {e}")
            return ""

    def _call_api_batch(self, prompts: list) -> list:
        """Call OpenAI API for a batch of prompts (sequential calls)."""
        results = []
        for prompt in prompts:
            result = self._call_api(prompt)
            results.append(result)
        return results

    def generate(
        self,
        graphs=None,  # Not used
        input_ids=None,  # Not used directly
        attention_mask=None,  # Not used
        is_mol_token=None,  # Not used
        do_sample=False,
        num_beams=1,  # Not used for API
        max_length=128,
        min_length=1,  # Not used for API
        top_p=0.9,
        repetition_penalty=1.0,  # Not used for API
        length_penalty=1.0,  # Not used for API
        num_captions=1,  # Not used for API
        temperature=None,
        output_attentions=False,  # Not used for API
        prompt_texts=None,  # Raw prompt texts - REQUIRED
        return_scores=False,  # Not supported for API
    ):
        """
        Generate text using OpenAI API.

        Args:
            prompt_texts: List of prompt strings (REQUIRED)
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (0 for deterministic)
            top_p: Top-p sampling parameter

        Returns:
            GenerateOutput with predictions
        """
        if prompt_texts is None:
            raise ValueError("prompt_texts must be provided for GPT-4 generation")

        # Update generation parameters if provided
        if temperature is not None:
            self.temperature = temperature
        if max_length is not None:
            self.max_tokens = max_length
        if top_p is not None:
            self.top_p = top_p

        # Call API for each prompt
        predictions = self._call_api_batch(prompt_texts)

        # Create output object compatible with blip2_stage3
        class GenerateOutput:
            pass

        outputs = GenerateOutput()
        outputs.predictions = predictions
        outputs.sequences = None
        outputs.attentions = None
        outputs.logits = None  # API doesn't return logits

        return outputs
