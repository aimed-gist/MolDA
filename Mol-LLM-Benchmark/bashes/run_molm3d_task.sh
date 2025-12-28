#!/bin/bash

# ============================================================================
# MoLM3D Single Task Runner
# Usage: bash run_molm3d_task.sh <task_name>
# Example: bash run_molm3d_task.sh bace
#          bash run_molm3d_task.sh smol-property_prediction-bbbp
# ============================================================================

set -e

# Check argument
if [ -z "$1" ]; then
    echo "Usage: bash run_molm3d_task.sh <task_name>"
    echo ""
    echo "Available tasks:"
    echo "  # Text2Mol / Mol2Text"
    echo "  chebi-20-text2mol"
    echo "  smol-molecule_generation"
    echo "  chebi-20-mol2text"
    echo "  smol-molecule_captioning"
    echo ""
    echo "  # Regression"
    echo "  smol-property_prediction-esol"
    echo "  smol-property_prediction-lipo"
    echo "  qm9_homo"
    echo "  qm9_lumo"
    echo "  qm9_homo_lumo_gap"
    echo "  aqsol-logS"
    echo ""
    echo "  # Reaction"
    echo "  forward_reaction_prediction"
    echo "  retrosynthesis"
    echo "  reagent_prediction"
    echo "  smol-forward_synthesis"
    echo "  smol-retrosynthesis"
    echo ""
    echo "  # Classification"
    echo "  bace"
    echo "  smol-property_prediction-bbbp"
    echo "  smol-property_prediction-clintox"
    echo "  smol-property_prediction-hiv"
    echo "  smol-property_prediction-sider"
    exit 1
fi

TASK_NAME="$1"
DATE_TAG=$(date +%Y%m%d)

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_CONFIG="${PROJECT_ROOT}/configs/data/default.yaml"
TEST_CONFIG="${PROJECT_ROOT}/configs/test_molm_3d.yaml"

# Settings
GPUS="'0,1,2,3'"
DATA_ROOT="/workspace/DATA/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_test_512_Truncation_indexed"
LOG_BASE="/workspace/Mol_DA/Mol-LLM-Benchmark/checkpoint/HJ_1223/logs/intrinsic_false"

echo "=============================================="
echo "MoLM3D Single Task Runner"
echo "=============================================="
echo "Task: ${TASK_NAME}"
echo "Date: ${DATE_TAG}"
echo ""

# Step 1: Update data/default.yaml - activate only the specified task
echo "[Step 1] Updating ${DATA_CONFIG}..."

# Create the new tasks section with only the specified task active
cat > "${DATA_CONFIG}" << EOF
raw_data_root: /workspace/Origin/Mol-LLM_Custom/dataset/real_train
# data_tag: 3.3M_0415
direct_data_root: null

tasks:
  # - chebi-20-text2mol
  # - smol-molecule_generation

  # - chebi-20-mol2text
  # - smol-molecule_captioning


  # - smol-property_prediction-esol
  # - smol-property_prediction-lipo
  # - qm9_homo
  # - qm9_lumo
  # - qm9_homo_lumo_gap

  # - forward_reaction_prediction
  # - retrosynthesis
  # - reagent_prediction
  # - smol-forward_synthesis
  # - smol-retrosynthesis

  # - aqsol-logS
#classification benchmarks
  # - bace
  # - smol-property_prediction-bbbp
  # - smol-property_prediction-clintox
  # - smol-property_prediction-hiv
  # - smol-property_prediction-sider

  - ${TASK_NAME}

EOF

echo "  Activated task: ${TASK_NAME}"

# Step 2: Update test_molm_3d.yaml - update logging_dir
echo "[Step 2] Updating ${TEST_CONFIG}..."

# Update logging_dir line
NEW_LOG_DIR="${LOG_BASE}/molm_3d_${DATE_TAG}_${TASK_NAME}"
sed -i "s|^logging_dir:.*|logging_dir: ${NEW_LOG_DIR}|" "${TEST_CONFIG}"

echo "  Logging dir: ${NEW_LOG_DIR}"

# Step 3: Show config summary
echo ""
echo "[Config Summary]"
echo "  Data config: ${DATA_CONFIG}"
echo "  Test config: ${TEST_CONFIG}"
echo "  GPUs: ${GPUS}"
echo "  Data root: ${DATA_ROOT}"
echo ""

# Step 4: Run the test
echo "[Step 3] Running MoLM3D test..."
echo "=============================================="

cd "${PROJECT_ROOT}"

export TOKENIZERS_PARALLELISM=false

python stage3.py \
    --config-name=test_molm_3d \
    trainer.devices=${GPUS} \
    mode=test \
    filename=molm_3d_${TASK_NAME} \
    +data.direct_data_root=${DATA_ROOT} \
    trainer.skip_sanity_check=false \
    +return_scores=true

echo ""
echo "=============================================="
echo "Test completed for task: ${TASK_NAME}"
echo "Results saved to: ${NEW_LOG_DIR}"
echo "=============================================="
