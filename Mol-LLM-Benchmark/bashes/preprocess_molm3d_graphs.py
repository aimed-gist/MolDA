#!/usr/bin/env python3
"""
MoLM3D 3D Graph Preprocessing Script

데이터셋의 모든 샘플에 대해 3D 그래프를 미리 생성하여 캐시 파일로 저장합니다.
이를 통해 추론 시 SMILES → 3D Graph 변환 시간을 제거합니다.

Usage:
    python bashes/preprocess_molm3d_graphs.py --data_root /path/to/dataset [--num_workers 8] [--timeout 30]
"""

import os
import sys
import re
import argparse
import torch
import numpy as np
import time
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pebble import ProcessPool
from tqdm import tqdm
from datasets import load_from_disk
import selfies as sf
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import distance_matrix

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# UniMol dictionary path
UNIMOL_DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model", "molm_3d", "unimol_dict.txt"
)


def load_unimol_dictionary():
    """UniMol dictionary 로드"""
    from unicore.data import Dictionary
    dictionary = Dictionary.load(UNIMOL_DICT_PATH)
    dictionary.add_symbol("[MASK]", is_special=True)
    return dictionary


def extract_selfies_from_input(input_mol_string):
    """input_mol_string에서 SELFIES 추출"""
    if not input_mol_string:
        return None
    # <SELFIES> ... </SELFIES> 패턴 매칭
    match = re.search(r'<SELFIES>\s*(.*?)\s*</SELFIES>', input_mol_string)
    if match:
        selfies = match.group(1).strip()
        # <None> 문자열은 실제 SELFIES가 아님 (text2mol 태스크 등)
        if selfies in ('<None>', 'None', '<none>', 'none', ''):
            return None
        return selfies
    # SELFIES 태그 없이 직접 SELFIES인 경우
    if input_mol_string.strip().startswith('['):
        return input_mol_string.strip()
    return None


def smiles2graph_2d(smiles, dictionary):
    """SMILES를 2D 좌표 기반 그래프로 변환 (빠른 fallback용)"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "invalid_smiles_2d"

        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if not atoms:
            return None, "no_atoms_2d"

        # 2D 좌표 생성 (매우 빠름, 실패 거의 없음)
        AllChem.Compute2DCoords(mol)
        coordinates = mol.GetConformer().GetPositions().astype(np.float32)

        atoms = np.asarray(atoms)

        # atom vectors (as numpy array)
        atom_vec = dictionary.vec_index(atoms).astype(np.int64)

        # normalize coordinates
        coordinates = coordinates - coordinates.mean(axis=0)

        # add special tokens (BOS, EOS)
        atom_vec = np.concatenate([
            np.array([dictionary.bos()], dtype=np.int64),
            atom_vec,
            np.array([dictionary.eos()], dtype=np.int64)
        ])
        coordinates = np.concatenate([
            np.zeros((1, 3), dtype=np.float32),
            coordinates,
            np.zeros((1, 3), dtype=np.float32)
        ], axis=0)

        # edge types and distances (as numpy arrays)
        dict_len = len(dictionary)
        edge_type = atom_vec.reshape(-1, 1) * dict_len + atom_vec.reshape(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)

        return (atom_vec, dist, edge_type), None

    except Exception as e:
        return None, f"exception_2d:{str(e)[:50]}"


def smiles2graph_openbabel(smiles, dictionary):
    """OpenBabel을 사용한 3D 좌표 생성 (RDKit 실패 시 fallback)"""
    try:
        from openbabel import pybel

        # SMILES → 3D 구조 생성
        mol = pybel.readstring("smi", smiles)
        mol.addh()  # 수소 추가
        mol.make3D()  # 3D 좌표 생성

        # 원자 정보 추출 (수소 제외)
        atoms = []
        coordinates = []
        for atom in mol.atoms:
            symbol = pybel.ob.GetSymbol(atom.atomicnum)
            if symbol != 'H':
                atoms.append(symbol)
                coordinates.append(atom.coords)

        if not atoms:
            return None, "no_atoms_openbabel"

        atoms = np.asarray(atoms)
        coordinates = np.array(coordinates, dtype=np.float32)

        # atom vectors
        atom_vec = dictionary.vec_index(atoms).astype(np.int64)

        # normalize coordinates
        coordinates = coordinates - coordinates.mean(axis=0)

        # add special tokens (BOS, EOS)
        atom_vec = np.concatenate([
            np.array([dictionary.bos()], dtype=np.int64),
            atom_vec,
            np.array([dictionary.eos()], dtype=np.int64)
        ])
        coordinates = np.concatenate([
            np.zeros((1, 3), dtype=np.float32),
            coordinates,
            np.zeros((1, 3), dtype=np.float32)
        ], axis=0)

        # edge types and distances
        dict_len = len(dictionary)
        edge_type = atom_vec.reshape(-1, 1) * dict_len + atom_vec.reshape(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)

        return (atom_vec, dist, edge_type), None

    except Exception as e:
        return None, f"exception_openbabel:{str(e)[:50]}"


def smiles2graph_3d(smiles, dictionary):
    """SMILES를 3D 그래프로 변환 (inference.ipynb 기반)

    Returns numpy arrays instead of torch tensors to avoid multiprocessing mmap issues.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "invalid_smiles"

        mol = AllChem.AddHs(mol)
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]

        if (np.asarray(atoms) == 'H').all():
            return None, "hydrogen_only"

        # 3D 좌표 생성 (maxAttempts 줄임)
        res = AllChem.EmbedMolecule(mol, maxAttempts=500)
        if res == 0:
            try:
                AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            except:
                pass
            coordinates = mol.GetConformer().GetPositions()
        elif res == -1:
            mol_tmp = Chem.MolFromSmiles(smiles)
            if mol_tmp is None:
                return None, "embed_failed_invalid_smiles"
            res2 = AllChem.EmbedMolecule(mol_tmp, maxAttempts=500)
            if res2 != 0:
                return None, "embed_failed"
            mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
            try:
                AllChem.MMFFOptimizeMolecule(mol_tmp, maxIters=200)
            except:
                pass
            try:
                coordinates = mol_tmp.GetConformer().GetPositions()
            except:
                return None, "no_conformer"
        else:
            return None, f"embed_error_{res}"

        coordinates = coordinates.astype(np.float32)

        if len(atoms) != len(coordinates):
            return None, "atom_coord_mismatch"

        atoms = np.asarray(atoms)

        # 수소 제거
        mask_hydrogen = atoms != "H"
        if sum(mask_hydrogen) > 0:
            atoms = atoms[mask_hydrogen]
            coordinates = coordinates[mask_hydrogen]

        # atom vectors (as numpy array)
        atom_vec = dictionary.vec_index(atoms).astype(np.int64)

        # normalize coordinates
        coordinates = coordinates - coordinates.mean(axis=0)

        # add special tokens (BOS, EOS)
        atom_vec = np.concatenate([
            np.array([dictionary.bos()], dtype=np.int64),
            atom_vec,
            np.array([dictionary.eos()], dtype=np.int64)
        ])
        coordinates = np.concatenate([
            np.zeros((1, 3), dtype=np.float32),
            coordinates,
            np.zeros((1, 3), dtype=np.float32)
        ], axis=0)

        # edge types and distances (as numpy arrays)
        dict_len = len(dictionary)
        edge_type = atom_vec.reshape(-1, 1) * dict_len + atom_vec.reshape(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)

        # Return as numpy arrays (will be converted to tensors when loading)
        return (atom_vec, dist, edge_type), None

    except Exception as e:
        return None, f"exception:{str(e)[:50]}"


def process_single_sample(args):
    """단일 샘플 처리 함수 - pebble ProcessPool용"""
    idx, input_mol_string, dict_path = args

    # 각 프로세스에서 dictionary 로드
    from unicore.data import Dictionary
    dictionary = Dictionary.load(dict_path)
    dictionary.add_symbol("[MASK]", is_special=True)

    try:
        # SELFIES 추출
        selfies_str = extract_selfies_from_input(input_mol_string)
        if not selfies_str:
            # text2mol 태스크 등 입력에 분자가 없는 경우
            return idx, None, "no_input_molecule", None

        # SELFIES → SMILES 변환
        smiles = sf.decoder(selfies_str)
        if not smiles:
            return idx, None, "selfies_decode_failed", selfies_str

        # 3D 그래프 생성 시도 (RDKit)
        graph, error = smiles2graph_3d(smiles, dictionary)

        if graph is not None:
            return idx, graph, None, None
        else:
            # RDKit 3D 실패 시 OpenBabel 3D 시도
            graph_ob, error_ob = smiles2graph_openbabel(smiles, dictionary)
            if graph_ob is not None:
                return idx, graph_ob, "fallback_openbabel", smiles
            else:
                # OpenBabel도 실패 시 RDKit 2D fallback
                graph_2d, error_2d = smiles2graph_2d(smiles, dictionary)
                if graph_2d is not None:
                    return idx, graph_2d, "fallback_2d", smiles
                else:
                    return idx, None, f"{error}+{error_ob}+{error_2d}", smiles

    except Exception as e:
        return idx, None, f"exception:{str(e)[:50]}", None


def main():
    parser = argparse.ArgumentParser(description='Preprocess MoLM3D 3D graphs')
    parser.add_argument('--data_root', type=str, default="/workspace/DATA/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_test_512_Truncation_indexed",
                        help='Path to the dataset (direct_data_root)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of worker processes (default: 16)')
    parser.add_argument('--output', type=str, default="/workspace/DATA/GSAI-ML-LLaDA-8B-Instruct_string+graph_q32_test_512_Truncation_indexed/molm3d_graphs.pt",
                        help='Output path (default: {data_root}_molm3d_graphs.pt)')
    parser.add_argument('--timeout', type=int, default=30,
                        help='Timeout per molecule in seconds (default: 30)')
    args = parser.parse_args()

    # 출력 경로 결정
    if args.output:
        output_path = args.output
    else:
        output_path = args.data_root.rstrip('/') + '_molm3d_graphs.pt'

    # 로그 파일 경로
    log_path = output_path.replace('.pt', '_failed.log')

    print(f"Loading dataset from: {args.data_root}")
    dataset = load_from_disk(args.data_root)
    print(f"Dataset size: {len(dataset)}")

    # 워커 수 결정
    num_workers = args.num_workers or 16
    print(f"Using {num_workers} workers")
    print(f"Timeout per molecule: {args.timeout}s")

    # 처리할 샘플 준비
    print("Preparing samples...")
    samples = []
    for i, sample in enumerate(tqdm(dataset, desc="Collecting samples")):
        idx = sample.get('idx', i)
        input_mol_string = sample.get('input_mol_string', '')
        samples.append((idx, input_mol_string, UNIMOL_DICT_PATH))

    # pebble ProcessPool로 그래프 생성 (진정한 프로세스 레벨 타임아웃)
    print("Generating 3D graphs with pebble ProcessPool...")
    graph_cache = {}
    failed_samples = []
    fallback_samples = []
    timeout_samples = []  # 타임아웃된 샘플 정보 저장
    timeout_count = 0
    current_position = 0  # 현재 처리 중인 위치 추적

    with ProcessPool(max_workers=num_workers) as pool:
        future = pool.map(process_single_sample, samples, timeout=args.timeout)
        iterator = future.result()

        with tqdm(total=len(samples), desc="Processing") as pbar:
            while True:
                try:
                    result = next(iterator)
                    idx, graph, error, detail = result

                    if graph is not None:
                        graph_cache[idx] = graph
                        if error and "fallback" in error:
                            fallback_samples.append((idx, error, detail))
                    else:
                        failed_samples.append((idx, error, detail))

                    current_position += 1
                    pbar.update(1)

                except StopIteration:
                    break
                except FuturesTimeoutError:
                    # 타임아웃 발생 - 해당 샘플 정보 기록
                    if current_position < len(samples):
                        timed_out_sample = samples[current_position]
                        idx, input_mol_string, _ = timed_out_sample
                        selfies_str = extract_selfies_from_input(input_mol_string)
                        try:
                            smiles = sf.decoder(selfies_str) if selfies_str else None
                        except:
                            smiles = None
                        timeout_samples.append((idx, selfies_str, smiles))

                    timeout_count += 1
                    current_position += 1
                    pbar.update(1)
                except Exception as e:
                    failed_samples.append((None, f"pool_exception:{str(e)[:50]}", None))
                    current_position += 1
                    pbar.update(1)

    print(f"\nSuccessfully processed: {len(graph_cache)}/{len(samples)}")
    print(f"  - 3D coordinates: {len(graph_cache) - len(fallback_samples)}")
    print(f"  - 2D fallback: {len(fallback_samples)}")
    print(f"Timeouts (will use runtime fallback): {timeout_count}")
    print(f"Failed: {len(failed_samples)}")

    # 실패 로그 저장
    if failed_samples or fallback_samples or timeout_samples:
        print(f"Writing log to: {log_path}")
        with open(log_path, 'w') as f:
            f.write(f"# Processing Summary\n")
            f.write(f"# Total samples: {len(samples)}\n")
            f.write(f"# Successfully processed: {len(graph_cache)}\n")
            f.write(f"# - 3D coordinates: {len(graph_cache) - len(fallback_samples)}\n")
            f.write(f"# - 2D fallback: {len(fallback_samples)}\n")
            f.write(f"# Timeouts: {timeout_count}\n")
            f.write(f"# Failed: {len(failed_samples)}\n\n")

            if failed_samples:
                # 에러 타입별 통계
                error_counts = {}
                for idx, error, detail in failed_samples:
                    error_counts[error] = error_counts.get(error, 0) + 1

                f.write("## Error Statistics:\n")
                for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                    f.write(f"  {error}: {count}\n")
                f.write("\n## Failed Samples:\n")

                for idx, error, detail in failed_samples:
                    f.write(f"{idx} | {error} | {detail}\n")

                # 콘솔에도 에러 통계 출력
                print("\nError Statistics:")
                for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                    print(f"  {error}: {count}")

            if timeout_samples:
                f.write("\n## Timeout Samples (idx | SELFIES | SMILES):\n")
                for idx, selfies, smiles in timeout_samples:
                    selfies_display = selfies if selfies else "None"
                    smiles_display = smiles if smiles else "None"
                    f.write(f"{idx} | {selfies_display} | {smiles_display}\n")
                print(f"\nTimeout samples logged: {len(timeout_samples)}")

            if fallback_samples:
                f.write("\n## 2D Fallback Samples:\n")
                for idx, error, detail in fallback_samples:
                    f.write(f"{idx} | {error} | {detail}\n")

    # 캐시 저장
    print(f"\nSaving cache to: {output_path}")
    torch.save(graph_cache, output_path)

    # 파일 크기 확인
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"Cache file size: {file_size:.2f} MB")
    print("Done!")


if __name__ == '__main__':
    main()
