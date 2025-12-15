import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModelForCausalLM

from .config import BASE_MODELS


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device


def load_tokenizer_and_model(model_name, base_model=None, device=None, use_device_map=False):
    """
    Load LlaSMol tokenizer and model with LoRA adapter merged.

    Args:
        model_name: HuggingFace adapter name (e.g., 'osunlp/LlaSMol-Mistral-7B')
        base_model: Base model name (e.g., 'mistralai/Mistral-7B-v0.1')
        device: Target device (if None, auto-detect)
        use_device_map: If True, use device_map="auto" (not compatible with DDP)
    """
    if base_model is None:
        if model_name in BASE_MODELS:
            base_model = BASE_MODELS[model_name]
    assert base_model is not None, "Please assign the corresponding base model to the argument 'base_model'."

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = 'left'
    # Use eos_token as pad_token (Mistral doesn't have a dedicated pad token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.sep_token = '<unk>'
    tokenizer.cls_token = '<unk>'
    tokenizer.mask_token = '<unk>'

    if device is None:
        device = get_device()

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if device == "cuda":
        if use_device_map:
            # Use device_map for single-GPU or model parallelism
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=dtype,
                device_map="auto",
            )
        else:
            # Load without device_map for DDP compatibility
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=dtype,
            )

        model = PeftModelForCausalLM.from_pretrained(
            model,
            model_name,
            torch_dtype=dtype,
        )
    else:
        raise NotImplementedError("No implementation for loading model on CPU yet.")

    # Merge LoRA weights into base model
    model = model.merge_and_unload()
    print(f"[LlaSMol] LoRA adapter merged successfully")

    # Update config with tokenizer IDs
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return tokenizer, model
