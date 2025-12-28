#!/usr/bin/env python3
"""
원본 테스트 데이터셋에 idx 컬럼을 추가하는 일회성 스크립트
실행: python bashes/add_idx_to_dataset.py
"""

from datasets import load_from_disk

# INPUT_PATH = "/workspace/DATA/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_test_3.3M_0415_verified_filtered_512"
# OUTPUT_PATH = "/workspace/DATA/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_test_3.3M_0415_verified_filtered_512_indexed"

# direct_data_root="/workspace/Origin/Mol_llm_Origin/data/mol-llm_testset"
direct_data_root="/workspace/DATA/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_test_512_Truncation"
INPUT_PATH = direct_data_root
OUTPUT_PATH = direct_data_root + "_indexed"

def main():
    print(f"Loading dataset from: {INPUT_PATH}")
    dataset = load_from_disk(INPUT_PATH)
    print(f"Original dataset size: {len(dataset)}")
    print(f"Original columns: {dataset.column_names}")

    # idx 컬럼 추가
    print("Adding idx column...")
    dataset = dataset.map(lambda x, idx: {"idx": idx, **x}, with_indices=True)

    print(f"New columns: {dataset.column_names}")
    print(f"Sample 0: idx={dataset[0]['idx']}, task={dataset[0]['task']}")
    print(f"Sample 100: idx={dataset[100]['idx']}, task={dataset[100]['task']}")

    # 저장
    print(f"Saving to: {OUTPUT_PATH}")
    dataset.save_to_disk(OUTPUT_PATH)
    print("Done!")

    # 검증
    print("\nVerifying saved dataset...")
    loaded = load_from_disk(OUTPUT_PATH)
    print(f"Loaded dataset size: {len(loaded)}")
    print(f"Columns: {loaded.column_names}")
    assert "idx" in loaded.column_names, "idx column not found!"
    assert loaded[0]["idx"] == 0, "idx value mismatch!"
    print("Verification passed!")

if __name__ == "__main__":
    main()
