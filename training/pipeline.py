"""
Generic Bioactivity Pipeline Module.

Provides reusable functions for the complete ML pipeline:
    1. Extract bioactivity data from ChEMBL 37
    2. Process and clean data (canonicalize, dedup, label, split)
    3. Featurize compounds (Morgan FP + descriptors)
    4. Train XGBoost model with Optuna hyperparameter tuning
    5. Run biological benchmark validation

Each function is parameterized by a target configuration dict.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Shared constants
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
N_CV_FOLDS = 5
N_OPTUNA_TRIALS = 50
RANDOM_SEED = 42
TEST_FRACTION = 0.2
CONFIDENCE_SCORE_MIN = 8

logger = logging.getLogger(__name__)


def setup_logging(target_name: str, step: str) -> None:
    """Configure logging for a pipeline step.

    Args:
        target_name: Target short name (e.g., 'amylase').
        step: Pipeline step (e.g., 'extraction', 'processing').
    """
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remove any existing handlers to avoid duplicate output
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f"{target_name}_{step}.log"),
        ],
        force=True,
    )


# ── Step 1: Extraction ─────────────────────────────────────────────────────────


def extract_activities(config: dict) -> pd.DataFrame:
    """Extract bioactivity data from ChEMBL for a given target.

    Args:
        config: Target configuration dict with keys:
            - chembl_id: ChEMBL target ID
            - name: Target name
            - activity_types: List of activity types (e.g., ['IC50', 'Ki'])

    Returns:
        DataFrame with raw bioactivity records.
    """
    from chembl_webresource_client.settings import Settings
    Settings.Instance().MAX_LIMIT = 1000
    from chembl_webresource_client.new_client import new_client

    target_ids = config["chembl_id"]
    if isinstance(target_ids, str):
        target_ids = [target_ids]
    target_name = config["name"]
    activity_types = config.get("activity_types", ["IC50", "Ki"])

    setup_logging(target_name.lower(), "extraction")
    logger.info(f"Starting extraction for {target_name} ({target_ids})")

    activity = new_client.activity
    all_records = []
    for target_id in target_ids:
        logger.info(f"Fetching records for {target_id}...")
        results = activity.filter(
            target_chembl_id=target_id,
            standard_type__in=activity_types,
            target_confidence_score__gte=CONFIDENCE_SCORE_MIN,
        )
        records = list(results)
        logger.info(f"  {target_id}: {len(records)} records")
        all_records.extend(records)

    records = all_records
    logger.info(f"Total retrieved: {len(records)} raw records")

    if not records:
        logger.error("No records retrieved. Check network or ChEMBL API availability.")
        return pd.DataFrame()

    df = pd.DataFrame(records)

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

    # Filter: only human/porcine (for pooled targets like amylase)
    if "target_organism" in df.columns:
        allowed_organisms = {"Homo sapiens", "Sus scrofa"}
        before_count = len(df)
        df = df[df["target_organism"].isin(allowed_organisms)]
        logger.info(f"Dropped {before_count - len(df)} records from non-human/porcine organisms")

    # Filter: remove records with data validity issues
    if "data_validity_comment" in df.columns:
        before_count = len(df)
        df = df[df["data_validity_comment"].isna()]
        logger.info(f"Dropped {before_count - len(df)} records with data validity issues")

    logger.info(f"Final dataset: {len(df)} records")
    logger.info(f"  Unique compounds: {df['molecule_chembl_id'].nunique()}")

    return df


def save_raw_dataset(df: pd.DataFrame, config: dict) -> Path:
    """Save raw extraction with metadata.

    Args:
        df: Raw bioactivity DataFrame.
        config: Target configuration dict.

    Returns:
        Path to the saved CSV file.
    """
    target_name = config["name"].lower()
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{target_name}_raw.csv"
    metadata = {
        "target": config["name"],
        "chembl_id": config["chembl_id"],
        "extraction_date": datetime.now().isoformat(),
        "activity_types": config.get("activity_types", ["IC50", "Ki"]),
        "confidence_score_min": CONFIDENCE_SCORE_MIN,
        "total_records": len(df),
        "unique_compounds": df["molecule_chembl_id"].nunique() if len(df) > 0 else 0,
    }

    metadata_path = output_dir / f"{target_name}_raw_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    df.to_csv(output_path, index=False)
    logger.info(f"Raw dataset saved to {output_path}")
    return output_path


def run_extraction(config: dict) -> None:
    """Run the full extraction step for a target.

    Args:
        config: Target configuration dict.
    """
    setup_logging(config["name"].lower(), "extraction")
    logger.info("=" * 60)
    logger.info(f"{config['name']} Data Extraction Pipeline")
    logger.info("=" * 60)

    df = extract_activities(config)
    if df.empty:
        logger.error("Extraction failed — no data retrieved.")
        return
    save_raw_dataset(df, config)
    logger.info("Extraction complete.")


# ── Step 2: Processing ──────────────────────────────────────────────────────────


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
        Scaffold SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)
    except Exception:
        return smiles


def scaffold_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset by Murcko scaffolds to avoid data leakage.

    Args:
        df: DataFrame with 'canonical_smiles' column.
        test_fraction: Fraction for test set.
        seed: Random seed.

    Returns:
        Tuple of (train_df, test_df).
    """
    df = df.copy()
    df["scaffold"] = df["canonical_smiles"].apply(get_murcko_scaffold)

    scaffold_groups = df.groupby("scaffold").size().reset_index(name="count")
    scaffold_groups = scaffold_groups.sort_values("count", ascending=False)

    rng = np.random.default_rng(seed)
    scaffolds = list(scaffold_groups["scaffold"].values)
    rng.shuffle(scaffolds)

    test_size_target = int(len(df) * test_fraction)
    test_scaffolds = set()
    test_count = 0

    for scaffold in scaffolds:
        scaffold_size = len(df[df["scaffold"] == scaffold])
        if test_count + scaffold_size <= test_size_target * 1.1:
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


def run_processing(config: dict) -> None:
    """Run the full data processing pipeline for a target.

    Args:
        config: Target configuration dict with keys:
            - name: Target name
            - activity_threshold_nM: Activity cutoff in nM
    """
    target_name = config["name"].lower()
    threshold = config["activity_threshold_nM"]

    setup_logging(target_name, "processing")
    logger.info("=" * 60)
    logger.info(f"{config['name']} Data Processing Pipeline")
    logger.info("=" * 60)

    raw_path = Path(f"data/raw/{target_name}_raw.csv")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        logger.error(f"Raw data not found at {raw_path}. Run extraction first.")
        return

    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} raw records")

    # Canonicalize SMILES
    logger.info("Canonicalizing SMILES...")
    df["canonical_smiles"] = df["canonical_smiles"].apply(canonicalize_smiles)
    before_count = len(df)
    df = df.dropna(subset=["canonical_smiles"])
    logger.info(f"Dropped {before_count - len(df)} records with invalid SMILES")

    # Remove exact duplicates
    before_count = len(df)
    df = df.drop_duplicates(subset=["canonical_smiles", "standard_type", "standard_value"])
    logger.info(f"Removed {before_count - len(df)} exact duplicate records")

    # Aggregate conflicting measurements (median)
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

    # Apply activity threshold
    agg_df["active"] = (agg_df["standard_value_median"] <= threshold).astype(int)

    # Deduplicate to one record per compound
    logger.info("Deduplicating to one record per compound...")
    agg_df = agg_df.sort_values("standard_value_count", ascending=False)
    final_df = agg_df.drop_duplicates(subset=["canonical_smiles"], keep="first")

    active_count = final_df["active"].sum()
    inactive_count = len(final_df) - active_count
    logger.info(f"Final dataset: {len(final_df)} compounds")
    logger.info(f"  Active (≤ {threshold} nM): {active_count}")
    logger.info(f"  Inactive (> {threshold} nM): {inactive_count}")
    logger.info(f"  Ratio: {active_count / len(final_df) * 100:.1f}% active")

    # Scaffold split
    train_df, test_df = scaffold_split(final_df)

    # Save outputs
    final_df.to_csv(output_dir / f"{target_name}_processed.csv", index=False)
    train_df.to_csv(output_dir / f"{target_name}_train.csv", index=False)
    test_df.to_csv(output_dir / f"{target_name}_test.csv", index=False)

    metadata = {
        "target": config["name"],
        "chembl_id": config["chembl_id"],
        "processing_date": datetime.now().isoformat(),
        "activity_threshold_nM": threshold,
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
    metadata_path = output_dir / f"{target_name}_processed_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Processing complete.")


# ── Step 3: Featurization ───────────────────────────────────────────────────────


def compute_morgan_fingerprint(smiles: str) -> np.ndarray | None:
    """Compute Morgan fingerprint for a SMILES string.

    Args:
        smiles: Canonical SMILES.

    Returns:
        Numpy array of shape (2048,) or None if invalid.
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
        Dictionary of descriptor values or None if invalid.
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

    assert X_fp.shape == (len(valid_indices), MORGAN_NBITS), \
        f"Unexpected fingerprint shape: {X_fp.shape}"
    assert not np.any(np.isnan(X_fp)), "NaN detected in fingerprints"
    assert len(y) == len(X_fp), "Label/feature count mismatch"

    logger.info(f"Fingerprint matrix shape: {X_fp.shape}")
    logger.info(f"Labels: {int(y.sum())} active, {int(len(y) - y.sum())} inactive")

    return X_fp, y, desc_df


def run_featurization(config: dict) -> None:
    """Run featurization for a target's train and test sets.

    Args:
        config: Target configuration dict.
    """
    target_name = config["name"].lower()
    setup_logging(target_name, "featurization")
    logger.info("=" * 60)
    logger.info(f"{config['name']} Feature Engineering")
    logger.info("=" * 60)

    processed_dir = Path("data/processed")

    for split in ["train", "test"]:
        input_path = processed_dir / f"{target_name}_{split}.csv"
        if not input_path.exists():
            logger.error(f"{input_path} not found. Run processing first.")
            continue

        X_fp, y, desc_df = featurize_dataset(input_path)

        np.save(processed_dir / f"{target_name}_{split}_fingerprints.npy", X_fp)
        np.save(processed_dir / f"{target_name}_{split}_labels.npy", y)
        desc_df.to_csv(processed_dir / f"{target_name}_{split}_descriptors.csv", index=False)

        logger.info(f"Saved {split} features: fingerprints={X_fp.shape}, labels={y.shape}")

    logger.info("Featurization complete.")


# ── Step 4: Training ────────────────────────────────────────────────────────────


def evaluate_model(model, X: np.ndarray, y: np.ndarray, label: str) -> dict:
    """Compute all required metrics.

    Args:
        model: Trained classifier.
        X: Feature matrix.
        y: True labels.
        label: Description for logging.

    Returns:
        Dictionary of metric name -> value.
    """
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y, y_prob),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "total_count": len(y),
    }

    logger.info(f"--- {label} metrics ---")
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.4f}")
        else:
            logger.info(f"  {name}: {value}")

    return metrics


def _optuna_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for XGBoost hyperparameter tuning."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]

        model.fit(X_train_cv, y_train_cv)
        y_prob = model.predict_proba(X_val_cv)[:, 1]
        scores.append(roc_auc_score(y_val_cv, y_prob))

    return np.mean(scores)


def run_training(config: dict) -> None:
    """Run the full model training pipeline for a target.

    Args:
        config: Target configuration dict.
    """
    target_name = config["name"].lower()
    setup_logging(target_name, "training")
    logger.info("=" * 60)
    logger.info(f"{config['name']} Model Training Pipeline")
    logger.info("=" * 60)

    processed_dir = Path("data/processed")
    model_dir = Path("models")
    results_dir = Path("results")
    for d in [model_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train = np.load(processed_dir / f"{target_name}_train_fingerprints.npy")
    y_train = np.load(processed_dir / f"{target_name}_train_labels.npy")
    X_test = np.load(processed_dir / f"{target_name}_test_fingerprints.npy")
    y_test = np.load(processed_dir / f"{target_name}_test_labels.npy")

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Train labels: {int(y_train.sum())} active, {int(len(y_train) - y_train.sum())} inactive")

    # Hyperparameter tuning
    logger.info(f"Starting Optuna hyperparameter search ({N_OPTUNA_TRIALS} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(lambda trial: _optuna_objective(trial, X_train, y_train), n_trials=N_OPTUNA_TRIALS)

    best_params = study.best_params
    logger.info(f"Best CV ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {best_params}")

    # Train final model
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    final_model = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    final_model.fit(X_train, y_train)

    # Evaluate
    train_metrics = evaluate_model(final_model, X_train, y_train, "Train")
    test_metrics = evaluate_model(final_model, X_test, y_test, "Test")

    # Save model
    model_path = model_dir / f"{target_name}_model.joblib"
    joblib.dump(final_model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save results
    results = {
        "target": config["name"],
        "training_date": datetime.now().isoformat(),
        "model_type": "XGBClassifier",
        "best_params": best_params,
        "best_cv_roc_auc": study.best_value,
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "n_cv_folds": N_CV_FOLDS,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    results_path = results_dir / f"{target_name}_training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info("Training complete.")


# ── Step 5: Benchmark Validation ────────────────────────────────────────────────


def smiles_to_fingerprint(smiles: str) -> np.ndarray | None:
    """Convert SMILES to Morgan fingerprint for benchmark scoring.

    Args:
        smiles: SMILES string.

    Returns:
        Fingerprint array or None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
    return np.array(fp)


def run_benchmark(config: dict) -> dict:
    """Score benchmark compounds and evaluate ranking.

    Args:
        config: Target configuration dict with 'benchmark_compounds' list.

    Returns:
        Dictionary with benchmark results and pass/fail status.
    """
    target_name = config["name"].lower()
    setup_logging(target_name, "benchmark")
    logger.info("=" * 60)
    logger.info(f"{config['name']} Biological Benchmark Validation")
    logger.info("=" * 60)

    model_dir = Path("models")
    results_dir = Path("results")
    benchmark_dir = Path("data/benchmarks")
    for d in [results_dir, benchmark_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{target_name}_model.joblib"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run training first.")
        return {"passed": False, "error": "Model not found"}

    model = joblib.load(model_path)
    logger.info(f"Loaded model from {model_path}")

    benchmark_compounds = config["benchmark_compounds"]

    # Score each compound
    results = []
    for compound in benchmark_compounds:
        fp = smiles_to_fingerprint(compound["smiles"])
        if fp is None:
            logger.warning(f"Invalid SMILES for {compound['name']}, skipping")
            continue

        prob = model.predict_proba(fp.reshape(1, -1))[0, 1]
        results.append({
            "name": compound["name"],
            "category": compound["category"],
            "probability": float(prob),
            "notes": compound.get("notes", ""),
        })
        logger.info(f"  {compound['name']:30s} | {compound['category']:10s} | P(active)={prob:.4f}")

    results_df = pd.DataFrame(results)

    # Compute category statistics
    category_stats = {}
    for category in ["potent", "weak", "unrelated"]:
        cat_probs = results_df[results_df["category"] == category]["probability"]
        if len(cat_probs) > 0:
            category_stats[category] = {
                "mean": float(cat_probs.mean()),
                "median": float(cat_probs.median()),
                "min": float(cat_probs.min()),
                "max": float(cat_probs.max()),
                "count": len(cat_probs),
            }
            logger.info(f"\n{category.upper()} — mean={cat_probs.mean():.4f}, "
                        f"median={cat_probs.median():.4f}, "
                        f"range=[{cat_probs.min():.4f}, {cat_probs.max():.4f}]")

    # Ranking tests
    potent_mean = category_stats.get("potent", {}).get("mean", 0)
    weak_mean = category_stats.get("weak", {}).get("mean", 0)
    unrelated_mean = category_stats.get("unrelated", {}).get("mean", 0)

    rank_potent_vs_weak = potent_mean > weak_mean
    rank_weak_vs_unrelated = weak_mean > unrelated_mean
    rank_potent_vs_unrelated = potent_mean > unrelated_mean

    potent_min = category_stats.get("potent", {}).get("min", 0)
    unrelated_max = category_stats.get("unrelated", {}).get("max", 1)
    strict_separation = potent_min > unrelated_max

    passed = rank_potent_vs_weak and rank_weak_vs_unrelated and rank_potent_vs_unrelated

    logger.info("\n" + "=" * 60)
    logger.info("RANKING TESTS:")
    logger.info(f"  potent_mean ({potent_mean:.4f}) > weak_mean ({weak_mean:.4f}): "
                f"{'PASS' if rank_potent_vs_weak else 'FAIL'}")
    logger.info(f"  weak_mean ({weak_mean:.4f}) > unrelated_mean ({unrelated_mean:.4f}): "
                f"{'PASS' if rank_weak_vs_unrelated else 'FAIL'}")
    logger.info(f"  potent_mean ({potent_mean:.4f}) > unrelated_mean ({unrelated_mean:.4f}): "
                f"{'PASS' if rank_potent_vs_unrelated else 'FAIL'}")
    logger.info(f"  Strict separation (potent_min > unrelated_max): "
                f"{'PASS' if strict_separation else 'FAIL'}")
    logger.info(f"\n  OVERALL: {'PASS' if passed else 'FAIL'}")
    logger.info("=" * 60)

    # Save results
    benchmark_results = {
        "target": config["name"],
        "benchmark_date": datetime.now().isoformat(),
        "compound_scores": results,
        "category_statistics": category_stats,
        "ranking_tests": {
            "potent_gt_weak": rank_potent_vs_weak,
            "weak_gt_unrelated": rank_weak_vs_unrelated,
            "potent_gt_unrelated": rank_potent_vs_unrelated,
            "strict_separation": strict_separation,
        },
        "passed": passed,
    }

    results_path = results_dir / f"{target_name}_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    results_df.to_csv(benchmark_dir / f"{target_name}_benchmark_panel.csv", index=False)

    return benchmark_results


# ── Full Pipeline Runner ────────────────────────────────────────────────────────


def run_full_pipeline(config: dict) -> dict:
    """Run the complete pipeline for a target (extract → process → featurize → train → benchmark).

    Args:
        config: Target configuration dict.

    Returns:
        Benchmark results dict.
    """
    target_name = config["name"]

    print("=" * 60)
    print(f"{target_name} FULL PIPELINE")
    print("=" * 60)

    print(f"\n[1/5] Extracting {target_name} data from ChEMBL...")
    run_extraction(config)

    print(f"\n[2/5] Processing and cleaning data...")
    run_processing(config)

    print(f"\n[3/5] Computing molecular fingerprints...")
    run_featurization(config)

    print(f"\n[4/5] Training {target_name} model...")
    run_training(config)

    print(f"\n[5/5] Running biological benchmark validation...")
    results = run_benchmark(config)

    print("\n" + "=" * 60)
    if results.get("passed"):
        print(f"PIPELINE COMPLETE — {target_name} model PASSED benchmark validation")
    else:
        print(f"PIPELINE COMPLETE — {target_name} model FAILED benchmark validation")
        print(f"Review results/{target_name.lower()}_benchmark_results.json for details.")
    print("=" * 60)

    return results
