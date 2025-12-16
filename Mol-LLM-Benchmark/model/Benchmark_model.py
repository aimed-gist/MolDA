"""
Pure LLM Benchmark Model
Unified wrapper for various LLM models (Galactica, Llama, Mistral, etc.)
Used for zero-shot benchmark testing with text-only molecular representations (SMILES/SELFIES)

No multimodal components (Q-Former, GNN) - Pure LLM only
"""

import logging
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
                **generation_settings,
            )

            # Extract predictions
            predictions = []
            for result in results:
                if result['output'] is not None:
                    predictions.append(result['output'][0] if result['output'] else "")
                else:
                    predictions.append("")

            # Create output object compatible with blip2_stage3
            class GenerateOutput:
                pass
            outputs = GenerateOutput()
            outputs.predictions = predictions
            outputs.sequences = None
            outputs.logits = None
            outputs.attentions = None

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
