"""
DPP4 Bioactivity Data Extraction from ChEMBL 37.

Extracts IC50/Ki activity data for DPP4 (CHEMBL284) with quality filters.
Outputs raw dataset to data/raw/dpp4_raw.csv with full metadata.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from chembl_webresource_client.new_client import new_client

# Configuration
TARGET_CHEMBL_ID = "CHEMBL284"
TARGET_NAME = "DPP4"
ACTIVITY_TYPES = ["IC50", "Ki"]
CONFIDENCE_SCORE_MIN = 8  # ChEMBL confidence ≥ 8 = direct single protein target
OUTPUT_DIR = Path("data/raw")
LOG_DIR = Path("logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "dpp4_extraction.log"),
    ],
)
logger = logging.getLogger(__name__)


def extract_dpp4_activities() -> pd.DataFrame:
    """Extract DPP4 bioactivity data from ChEMBL.

    Filters:
        - Target: CHEMBL284 (human DPP4)
        - Activity types: IC50, Ki
        - Confidence score >= 8
        - Only records with numeric standard_value
        - Only nM units (standardized)

    Returns:
        DataFrame with raw bioactivity records.
    """
    logger.info(f"Starting extraction for {TARGET_NAME} ({TARGET_CHEMBL_ID})")

    activity = new_client.activity
    results = activity.filter(
        target_chembl_id=TARGET_CHEMBL_ID,
        standard_type__in=ACTIVITY_TYPES,
        target_confidence_score__gte=CONFIDENCE_SCORE_MIN,
    )

    logger.info("Fetching records from ChEMBL API...")
    records = list(results)
    logger.info(f"Retrieved {len(records)} raw records")

    if not records:
        logger.error("No records retrieved. Check network or ChEMBL API availability.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Keep only essential columns
    columns_to_keep = [
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_value",
        "standard_units",
        "standard_relation",
        "pchembl_value",
        "assay_chembl_id",
        "assay_type",
        "target_chembl_id",
        "target_organism",
        "document_chembl_id",
        "data_validity_comment",
    ]
    available_cols = [c for c in columns_to_keep if c in df.columns]
    df = df[available_cols]

    # Filter: must have numeric standard_value
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    before_count = len(df)
    df = df.dropna(subset=["standard_value"])
    logger.info(f"Dropped {before_count - len(df)} records with non-numeric standard_value")

    # Filter: only nM units
    if "standard_units" in df.columns:
        before_count = len(df)
        df = df[df["standard_units"] == "nM"]
        logger.info(f"Dropped {before_count - len(df)} records with non-nM units")

    # Filter: must have canonical SMILES
    if "canonical_smiles" in df.columns:
        before_count = len(df)
        df = df.dropna(subset=["canonical_smiles"])
        df = df[df["canonical_smiles"].str.strip() != ""]
        logger.info(f"Dropped {before_count - len(df)} records without SMILES")

    # Filter: only human DPP4
    if "target_organism" in df.columns:
        before_count = len(df)
        df = df[df["target_organism"] == "Homo sapiens"]
        logger.info(f"Dropped {before_count - len(df)} non-human records")

    # Filter: remove records with data validity issues
    if "data_validity_comment" in df.columns:
        before_count = len(df)
        df = df[df["data_validity_comment"].isna()]
        logger.info(f"Dropped {before_count - len(df)} records with data validity issues")

    logger.info(f"Final dataset: {len(df)} records")
    logger.info(f"  IC50 records: {len(df[df['standard_type'] == 'IC50'])}")
    logger.info(f"  Ki records: {len(df[df['standard_type'] == 'Ki'])}")
    logger.info(f"  Unique compounds: {df['molecule_chembl_id'].nunique()}")

    return df


def save_raw_dataset(df: pd.DataFrame) -> Path:
    """Save raw extraction with metadata header.

    Args:
        df: Raw bioactivity DataFrame.

    Returns:
        Path to the saved CSV file.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "dpp4_raw.csv"

    # Save metadata alongside
    metadata = {
        "target": TARGET_NAME,
        "chembl_id": TARGET_CHEMBL_ID,
        "extraction_date": datetime.now().isoformat(),
        "activity_types": ACTIVITY_TYPES,
        "confidence_score_min": CONFIDENCE_SCORE_MIN,
        "total_records": len(df),
        "unique_compounds": df["molecule_chembl_id"].nunique() if len(df) > 0 else 0,
        "ic50_count": len(df[df["standard_type"] == "IC50"]) if len(df) > 0 else 0,
        "ki_count": len(df[df["standard_type"] == "Ki"]) if len(df) > 0 else 0,
    }

    metadata_path = OUTPUT_DIR / "dpp4_raw_metadata.json"
    import json
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    df.to_csv(output_path, index=False)
    logger.info(f"Raw dataset saved to {output_path}")

    return output_path


def main():
    """Run DPP4 data extraction pipeline."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DPP4 Data Extraction Pipeline")
    logger.info("=" * 60)

    df = extract_dpp4_activities()

    if df.empty:
        logger.error("Extraction failed — no data retrieved.")
        return

    output_path = save_raw_dataset(df)
    logger.info(f"Extraction complete. Output: {output_path}")


if __name__ == "__main__":
    main()
