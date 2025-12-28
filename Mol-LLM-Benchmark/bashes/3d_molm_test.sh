#!/bin/bash
source /workspace/binddm_env/etc/profile.d/conda.sh
conda activate mol_llm_env
export TOKENIZERS_PARALLELISM=false

# GPU settings
gpus="'0'"

# Data settings
direct_data_root="/workspace/Origin/Mol_llm_Origin/data/mol-llm_testset"

# Result filename
filename="3d_molm_test"

# 3D-MoLM checkpoint path
# Download from: https://huggingface.co/Sihangli/3D-MoLM/tree/main/generalist
ckpt_path_3dmolm="/workspace/Mol_DA_repo/Mol-LLM-Benchmark/model/3d_molm/checkpoints/generalist.ckpt"

echo "==============3D-MoLM Test==============="
echo "Make sure you have downloaded the checkpoint from HuggingFace:"
echo "  https://huggingface.co/Sihangli/3D-MoLM/tree/main/generalist"
echo "Place generalist.ckpt at: ${ckpt_path_3dmolm}"
echo ""

# Check if checkpoint exists
if [ ! -f "$ckpt_path_3dmolm" ]; then
    echo "ERROR: Checkpoint not found at ${ckpt_path_3dmolm}"
    echo "Please download the checkpoint first:"
    echo "  pip install huggingface_hub"
    echo "  huggingface-cli download Sihangli/3D-MoLM generalist/generalist.ckpt --local-dir /workspace/Mol_DA_repo/Mol-LLM-Benchmark/model/3d_molm/checkpoints"
    exit 1
fi

python stage3.py \
--config-name=test_3d_molm \
trainer.devices=$gpus \
mode=test \
filename=${filename} \
+data.direct_data_root=${direct_data_root} \
ckpt_path_3dmolm=${ckpt_path_3dmolm} \
trainer.skip_sanity_check=false
