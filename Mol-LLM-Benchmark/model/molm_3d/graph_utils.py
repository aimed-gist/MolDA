"""
3D-MoLM Graph Utilities
SMILES to 3D graph conversion for UniMol encoder
"""

import os
import torch
import numpy as np
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem import AllChem
from unicore.data import Dictionary


# Path to unimol dictionary
UNIMOL_DICT_PATH = os.path.join(
    os.path.dirname(__file__),
    'data_provider',
    'unimol_dict.txt'
)


def load_unimol_dictionary():
    """Load UniMol atom dictionary"""
    dictionary = Dictionary.load(UNIMOL_DICT_PATH)
    dictionary.add_symbol("[MASK]", is_special=True)
    return dictionary


def smiles2graph(smiles, dictionary):
    """
    Convert SMILES to 3D graph representation for UniMol.

    Args:
        smiles: SMILES string
        dictionary: UniMol Dictionary object

    Returns:
        Tuple of (atom_vec, dist, edge_type) or None if failed
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    if (np.asarray(atoms) == 'H').all():
        return None

    # Generate 3D coordinates
    res = AllChem.EmbedMolecule(mol)
    if res == 0:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass
        coordinates = mol.GetConformer().GetPositions()
    elif res == -1:
        mol_tmp = Chem.MolFromSmiles(smiles)
        AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000)
        mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
        try:
            AllChem.MMFFOptimizeMolecule(mol_tmp)
        except:
            pass
        try:
            coordinates = mol_tmp.GetConformer().GetPositions()
        except:
            # Fallback: use 2D coordinates if 3D fails
            AllChem.Compute2DCoords(mol)
            coordinates = mol.GetConformer().GetPositions()
            # Add zero z-coordinate
            coordinates = np.hstack([coordinates, np.zeros((len(coordinates), 1))])
    else:
        return None

    coordinates = coordinates.astype(np.float32)
    atoms = np.asarray(atoms)

    # Remove hydrogen atoms
    mask_hydrogen = atoms != "H"
    if sum(mask_hydrogen) > 0:
        atoms = atoms[mask_hydrogen]
        coordinates = coordinates[mask_hydrogen]

    # Atom vectors
    atom_vec = torch.from_numpy(dictionary.vec_index(atoms)).long()

    # Normalize coordinates
    coordinates = coordinates - coordinates.mean(axis=0)

    # Add special tokens (BOS/EOS)
    atom_vec = torch.cat([
        torch.LongTensor([dictionary.bos()]),
        atom_vec,
        torch.LongTensor([dictionary.eos()])
    ])
    coordinates = np.concatenate([
        np.zeros((1, 3)),
        coordinates,
        np.zeros((1, 3))
    ], axis=0)

    # Edge types
    edge_type = atom_vec.view(-1, 1) * len(dictionary) + atom_vec.view(1, -1)
    dist = distance_matrix(coordinates, coordinates).astype(np.float32)
    coordinates = torch.from_numpy(coordinates)
    dist = torch.from_numpy(dist)

    return atom_vec, dist, edge_type


def smiles2graph_2d(smiles, dictionary):
    """
    Convert SMILES to 2D graph representation (fast fallback).

    Uses 2D coordinates instead of 3D - much faster but less accurate.

    Args:
        smiles: SMILES string
        dictionary: UniMol Dictionary object

    Returns:
        Tuple of (atom_vec, dist, edge_type) or None if failed
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        if not atoms:
            return None

        # 2D coordinates (very fast, rarely fails)
        AllChem.Compute2DCoords(mol)
        coordinates = mol.GetConformer().GetPositions().astype(np.float32)

        atoms = np.asarray(atoms)

        # Atom vectors
        atom_vec = torch.from_numpy(dictionary.vec_index(atoms)).long()

        # Normalize coordinates
        coordinates = coordinates - coordinates.mean(axis=0)

        # Add special tokens (BOS/EOS)
        atom_vec = torch.cat([
            torch.LongTensor([dictionary.bos()]),
            atom_vec,
            torch.LongTensor([dictionary.eos()])
        ])
        coordinates = np.concatenate([
            np.zeros((1, 3), dtype=np.float32),
            coordinates,
            np.zeros((1, 3), dtype=np.float32)
        ], axis=0)

        # Edge types and distances
        edge_type = atom_vec.view(-1, 1) * len(dictionary) + atom_vec.view(1, -1)
        dist = distance_matrix(coordinates, coordinates).astype(np.float32)
        dist = torch.from_numpy(dist)

        return atom_vec, dist, edge_type

    except Exception:
        return None


def smiles2graph_with_timeout(smiles, dictionary, timeout=5):
    """
    Convert SMILES to 3D graph with timeout, falling back to 2D on failure.

    Args:
        smiles: SMILES string
        dictionary: UniMol Dictionary object
        timeout: Timeout in seconds for 3D generation (default: 5)

    Returns:
        Tuple of (atom_vec, dist, edge_type) or None if failed
    """
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError("3D graph generation timed out")

    # Try 3D first with timeout
    try:
        # Set up signal alarm (only works on Unix)
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)

        result = smiles2graph(smiles, dictionary)

        signal.alarm(0)  # Cancel alarm
        signal.signal(signal.SIGALRM, old_handler)  # Restore handler

        if result is not None:
            return result

    except TimeoutError:
        signal.alarm(0)
        print(f"[graph_utils] 3D generation timed out for SMILES: {smiles[:50]}..., using 2D fallback")
    except Exception as e:
        try:
            signal.alarm(0)
        except:
            pass
        print(f"[graph_utils] 3D generation failed: {str(e)[:50]}, using 2D fallback")

    # Fallback to 2D
    return smiles2graph_2d(smiles, dictionary)
