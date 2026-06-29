"""
DPP4 Data Processing Pipeline.

Takes raw ChEMBL extraction and produces a clean, deduplicated, labeled dataset
ready for feature engineering.

Steps:
    1. Canonicalize SMILES via RDKit
    2. Remove duplicates (same canonical SMILES + activity type)
    3. Handle conflicting measurements (median aggregation)
    4. Apply activity threshold (IC50 ≤ 10,000 nM = active)
    5. Generate train/test split (scaffold-based)
    6. Save processed dataset with metadata
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# Configuration
ACTIVITY_THRESHOLD_NM = 10_000  # IC50/Ki ≤ 10 µM (10,000 nM) = active
TEST_FRACTION = 0.2
RANDOM_SEED = 42

RAW_PATH = Path("data/raw/dpp4_raw.csv")
OUTPUT_DIR = Path("data/processed")
LOG_DIR = Path("logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "dpp4_processing.log"),
    ],
)
logger = logging.getLogger(__name__)


def canonicalize_smiles(smiles: str) -> str | None:
    """Convert SMILES to canonical form using RDKit.

    Args:
        smiles: Input SMILES string.

    Returns:
        Canonical SMILES or None if invalid.
    """
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def get_murcko_scaffold(smiles: str) -> str:
    """Get Murcko scaffold for scaffold-based splitting.

    Args:
        smiles: Canonical SMILES.

    Returns:
        Scaffold SMILES string, or the original SMILES if scaffolding fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return smiles


def scaffold_split(df: pd.DataFrame, test_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset by Murcko scaffolds to avoid data leakage.

    Compounds sharing the same scaffold go entirely into train OR test,
    never both. This prevents overoptimistic performance estimates from
    structurally similar compounds appearing in both sets.

    Args:
        df: DataFrame with 'canonical_smiles' column.
        test_fraction: Fraction of data for test set.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df).
    """
    logger.info("Computing Murcko scaffolds for splitting...")
    df = df.copy()
    df["scaffold"] = df["canonical_smiles"].apply(get_murcko_scaffold)

    # Group by scaffold, sort by size (largest first)
    scaffold_groups = df.groupby("scaffold").size().reset_index(name="count")
    scaffold_groups = scaffold_groups.sort_values("count", ascending=False)

    # Assign scaffolds to train/test to approximate the desired split
    rng = np.random.default_rng(seed)
    scaffolds = list(scaffold_groups["scaffold"].values)
    rng.shuffle(scaffolds)

    test_size_target = int(len(df) * test_fraction)
    test_scaffolds = set()
    test_count = 0

    for scaffold in scaffolds:
        scaffold_size = len(df[df["scaffold"] == scaffold])
        if test_count + scaffold_size <= test_size_target * 1.1:  # Allow 10% overshoot
            test_scaffolds.add(scaffold)
            test_count += scaffold_size
        if test_count >= test_size_target:
            break

    test_mask = df["scaffold"].isin(test_scaffolds)
    train_df = df[~test_mask].drop(columns=["scaffold"])
    test_df = df[test_mask].drop(columns=["scaffold"])

    logger.info(f"Scaffold split: train={len(train_df)}, test={len(test_df)} "
                f"({len(test_df) / len(df) * 100:.1f}% test)")

    return train_df, test_df


def process_dpp4_data() -> None:
    """Run the full DPP4 data processing pipeline."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DPP4 Data Processing Pipeline")
    logger.info("=" * 60)

    # Load raw data
    if not RAW_PATH.exists():
        logger.error(f"Raw data not found at {RAW_PATH}. Run extract_dpp4.py first.")
        return

    df = pd.read_csv(RAW_PATH)
    logger.info(f"Loaded {len(df)} raw records")

    # Step 1: Canonicalize SMILES
    logger.info("Canonicalizing SMILES...")
    df["canonical_smiles"] = df["canonical_smiles"].apply(canonicalize_smiles)
    before_count = len(df)
    df = df.dropna(subset=["canonical_smiles"])
    logger.info(f"Dropped {before_count - len(df)} records with invalid SMILES")

    # Step 2: Remove exact duplicates (same compound + same activity type)
    before_count = len(df)
    df = df.drop_duplicates(subset=["canonical_smiles", "standard_type", "standard_value"])
    logger.info(f"Removed {before_count - len(df)} exact duplicate records")

    # Step 3: Aggregate conflicting measurements
    # For the same compound and activity type, take the median value
    logger.info("Aggregating conflicting measurements (median)...")
    agg_df = (
        df.groupby(["canonical_smiles", "molecule_chembl_id", "standard_type"])
        .agg(
            standard_value_median=("standard_value", "median"),
            standard_value_count=("standard_value", "count"),
            standard_value_std=("standard_value", "std"),
            pchembl_value=("pchembl_value", "first"),
        )
        .reset_index()
    )
    logger.info(f"Aggregated to {len(agg_df)} unique compound-activity pairs")

    # Step 4: Apply activity threshold
    # Active: IC50/Ki ≤ 10,000 nM (10 µM)
    # Inactive: IC50/Ki > 10,000 nM
    agg_df["active"] = (agg_df["standard_value_median"] <= ACTIVITY_THRESHOLD_NM).astype(int)

    # If a compound has both IC50 and Ki, keep the one with more measurements
    # Then deduplicate to one record per compound
    logger.info("Deduplicating to one record per compound...")
    agg_df = agg_df.sort_values("standard_value_count", ascending=False)
    final_df = agg_df.drop_duplicates(subset=["canonical_smiles"], keep="first")

    active_count = final_df["active"].sum()
    inactive_count = len(final_df) - active_count
    logger.info(f"Final dataset: {len(final_df)} compounds")
    logger.info(f"  Active (≤ {ACTIVITY_THRESHOLD_NM} nM): {active_count}")
    logger.info(f"  Inactive (> {ACTIVITY_THRESHOLD_NM} nM): {inactive_count}")
    logger.info(f"  Ratio: {active_count / len(final_df) * 100:.1f}% active")

    # Step 5: Scaffold split
    train_df, test_df = scaffold_split(final_df, TEST_FRACTION, RANDOM_SEED)

    # Save outputs
    train_path = OUTPUT_DIR / "dpp4_train.csv"
    test_path = OUTPUT_DIR / "dpp4_test.csv"
    full_path = OUTPUT_DIR / "dpp4_processed.csv"

    final_df.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved: {full_path}, {train_path}, {test_path}")

    # Save metadata
    metadata = {
        "target": "DPP4",
        "chembl_id": "CHEMBL284",
        "processing_date": datetime.now().isoformat(),
        "activity_threshold_nM": ACTIVITY_THRESHOLD_NM,
        "total_compounds": len(final_df),
        "active_count": int(active_count),
        "inactive_count": int(inactive_count),
        "train_count": len(train_df),
        "test_count": len(test_df),
        "split_method": "scaffold",
        "test_fraction": TEST_FRACTION,
        "random_seed": RANDOM_SEED,
        "aggregation_method": "median",
    }
    metadata_path = OUTPUT_DIR / "dpp4_processed_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Metadata saved to {metadata_path}")
    logger.info("Processing complete.")


if __name__ == "__main__":
    process_dpp4_data()
