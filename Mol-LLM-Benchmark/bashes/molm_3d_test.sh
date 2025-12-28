#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# GPU settings
gpus="'0,1,2,3'"

# Data settings
direct_data_root="/workspace/DATA/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_test_512_Truncation_indexed"

# Result filename
filename="molm_3d_test"

echo "==============3D-MoLM Generalist Test==============="
python stage3.py \
--config-name=test_molm_3d \
trainer.devices=$gpus \
mode=test \
filename=${filename} \
+data.direct_data_root=${direct_data_root} \
trainer.skip_sanity_check=false \
+return_scores=false
