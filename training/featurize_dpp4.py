"""
DPP4 Feature Engineering.

Computes molecular fingerprints and descriptors for the processed DPP4 dataset.

Features:
    - Morgan fingerprints (radius=2, 2048 bits) — primary features
    - Molecular descriptors (MW, LogP, TPSA, HBD, HBA, RotBonds) — supplementary
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

# Configuration
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
PROCESSED_DIR = Path("data/processed")
LOG_DIR = Path("logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "dpp4_featurization.log"),
    ],
)
logger = logging.getLogger(__name__)


def compute_morgan_fingerprint(smiles: str) -> np.ndarray | None:
    """Compute Morgan fingerprint for a SMILES string.

    Args:
        smiles: Canonical SMILES.

    Returns:
        Numpy array of shape (2048,) or None if molecule is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
    return np.array(fp)


def compute_descriptors(smiles: str) -> dict | None:
    """Compute molecular descriptors for a SMILES string.

    Args:
        smiles: Canonical SMILES.

    Returns:
        Dictionary of descriptor values, or None if molecule is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "hbd": Descriptors.NumHDonors(mol),
        "hba": Descriptors.NumHAcceptors(mol),
        "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_atoms": mol.GetNumHeavyAtoms(),
        "num_rings": Descriptors.RingCount(mol),
    }


def featurize_dataset(input_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Featurize a processed dataset.

    Args:
        input_path: Path to processed CSV with 'canonical_smiles' and 'active' columns.

    Returns:
        Tuple of (fingerprints_array, labels_array, descriptor_dataframe).
    """
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} records from {input_path}")

    fingerprints = []
    descriptors = []
    valid_indices = []

    for idx, row in df.iterrows():
        smiles = row["canonical_smiles"]
        fp = compute_morgan_fingerprint(smiles)
        desc = compute_descriptors(smiles)

        if fp is not None and desc is not None:
            fingerprints.append(fp)
            descriptors.append(desc)
            valid_indices.append(idx)
        else:
            logger.warning(f"Failed to featurize: {smiles}")

    logger.info(f"Successfully featurized {len(valid_indices)}/{len(df)} compounds")

    X_fp = np.array(fingerprints)
    y = df.loc[valid_indices, "active"].values
    desc_df = pd.DataFrame(descriptors)

    # Validate
    assert X_fp.shape == (len(valid_indices), MORGAN_NBITS), \
        f"Unexpected fingerprint shape: {X_fp.shape}"
    assert not np.any(np.isnan(X_fp)), "NaN detected in fingerprints"
    assert len(y) == len(X_fp), "Label/feature count mismatch"

    logger.info(f"Fingerprint matrix shape: {X_fp.shape}")
    logger.info(f"Labels: {int(y.sum())} active, {int(len(y) - y.sum())} inactive")

    return X_fp, y, desc_df


def featurize_dpp4() -> None:
    """Run featurization for DPP4 train and test sets."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DPP4 Feature Engineering")
    logger.info("=" * 60)

    for split in ["train", "test"]:
        input_path = PROCESSED_DIR / f"dpp4_{split}.csv"
        if not input_path.exists():
            logger.error(f"{input_path} not found. Run process_dpp4.py first.")
            continue

        X_fp, y, desc_df = featurize_dataset(input_path)

        # Save as numpy arrays for fast loading during training
        np.save(PROCESSED_DIR / f"dpp4_{split}_fingerprints.npy", X_fp)
        np.save(PROCESSED_DIR / f"dpp4_{split}_labels.npy", y)
        desc_df.to_csv(PROCESSED_DIR / f"dpp4_{split}_descriptors.csv", index=False)

        logger.info(f"Saved {split} features: fingerprints={X_fp.shape}, labels={y.shape}")

    logger.info("Featurization complete.")


if __name__ == "__main__":
    featurize_dpp4()
