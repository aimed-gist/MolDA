#!/bin/bash
export TOKENIZERS_PARALLELISM=false

# GPU 설정
gpus="'0,1,2,3'"

# 데이터 설정
direct_data_root="/workspace/Origin/Mol_llm_Origin/data/mol-llm_testset"

# 결과 파일명
filename="galactica_test"

echo "==============Galactica 6.7B Test (HuggingFace)==============="
python stage3.py \
--config-name=test_galactica \
trainer.devices=$gpus \
mode=test \
filename=${filename} \
+data.direct_data_root=${direct_data_root} \
trainer.skip_sanity_check=false
