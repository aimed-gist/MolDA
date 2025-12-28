# 3D-MoLM Configuration

import os

# Default checkpoint paths
DEFAULT_CKPT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')

# Base models
BASE_MODELS = {
    '3d-molm-generalist': 'meta-llama/Llama-2-7b-hf',
}

# UniMol pretrained checkpoint path
# Download from: https://github.com/dptech-corp/Uni-Mol/releases/download/v0.1/mol_pre_all_h_220816.pt
UNIMOL_CKPT_PATH = os.path.join(DEFAULT_CKPT_DIR, 'mol_pre_all_h_220816.pt')

# Default UniMol settings
UNIMOL_DEFAULTS = {
    'unimol_encoder_layers': 15,
    'unimol_encoder_embed_dim': 512,
    'unimol_encoder_ffn_embed_dim': 2048,
    'unimol_encoder_attention_heads': 64,
    'unimol_activation_fn': 'gelu',
    'unimol_emb_dropout': 0.1,
    'unimol_dropout': 0.1,
    'unimol_attention_dropout': 0.1,
    'unimol_activation_dropout': 0.0,
    'unimol_max_seq_len': 512,
    'unimol_delta_pair_repr_norm_loss': -1.0,
    'unimol_max_atoms': 256,
    'unimol_ckpt_path': UNIMOL_CKPT_PATH,  # UniMol pretrained weights
}

# Q-Former settings
QFORMER_DEFAULTS = {
    'num_query_token': 8,  # 3D-MoLM 원본 체크포인트는 8개 query token 사용
    'cross_attention_freq': 2,
    'bert_name': 'scibert',
}

# LoRA settings
LORA_DEFAULTS = {
    'lora_r': 8,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
}
