#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# GPU 설정
gpus="'0,1,2,3'"

# 데이터 설정
direct_data_root="/workspace/DATA/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_test_3.3M_0415_verified_filtered_512_indexed"
# direct_data_root="/workspace/Origin/Mol_llm_Origin/data/mol-llm_testset_indexed"

# 결과 파일명
filename="chemdfm_test"

echo "==============ChemDFM-v1.0-13B Test (HuggingFace)==============="
python stage3.py \
--config-name=test_chemdfm \
trainer.devices=$gpus \
mode=test \
filename=${filename} \
+data.direct_data_root=${direct_data_root} \
trainer.skip_sanity_check=false
