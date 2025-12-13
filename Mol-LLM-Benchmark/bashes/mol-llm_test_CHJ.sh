export TOKENIZERS_PARALLELISM=false;
gpus="'0,1,2,3'" # 이게 실제 GPU 갯수 책임지는 곳,
# data_tag=3.3M_0415
direct_data_root="/workspace/Origin/Mol_llm_Origin/data/mol-llm_testset"
filename="HJChoi" # Replace with your actual filename
ckpt_path="/workspace/Origin/Mol_llm_Origin/checkpoint/mol-llm.ckpt" # Replace with your actual checkpoint path'"

echo "==============Executing task: Specific Task==============="
python stage3.py \
--config-name=test_CHJ \
trainer.devices=$gpus \
mode=test \
filename=${filename} \
data.direct_data_root=${direct_data_root} \
trainer=mistral7b_80gb \
trainer.skip_sanity_check=false \
ckpt_path=${ckpt_path}

