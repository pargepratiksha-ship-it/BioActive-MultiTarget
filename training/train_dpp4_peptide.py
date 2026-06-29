"""
Train DPP4 Peptide-Specific Model.

Uses curated peptide bioactivity data with sequence-based features to build
a model that genuinely predicts peptide DPP4 inhibition from IC50 data.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
)
from sklearn.preprocessing import StandardScaler
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))
from training.peptide_features import featurize_peptide, featurize_dataset, get_feature_names

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "peptides" / "dpp4_peptides.csv"
MODEL_OUTPUT = PROJECT_ROOT / "models" / "dpp4_peptide_model.joblib"
SCALER_OUTPUT = PROJECT_ROOT / "models" / "dpp4_peptide_scaler.joblib"
FEATURES_OUTPUT = PROJECT_ROOT / "data" / "processed" / "dpp4_peptide_features.npy"

# Activity threshold: IC50 <= 200 µM is considered active for DPP4 peptides
# This is more lenient than small molecules because food-derived peptides
# typically have weaker binding but are consumed in larger quantities
ACTIVITY_THRESHOLD_UM = 200.0


def load_and_prepare_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load peptide data and compute features."""
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"  Total peptides: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Assign binary labels based on IC50 threshold
    df["label"] = (df["ic50_um"] <= ACTIVITY_THRESHOLD_UM).astype(int)
    print(f"  Active (IC50 <= {ACTIVITY_THRESHOLD_UM} µM): {df['label'].sum()}")
    print(f"  Inactive (IC50 > {ACTIVITY_THRESHOLD_UM} µM): {(1 - df['label']).sum()}")

    # Featurize all sequences
    sequences = df["sequence"].tolist()
    print(f"\nFeaturizing {len(sequences)} peptides...")
    features_result = featurize_dataset(sequences, target="dpp4")

    if features_result is None:
        raise ValueError("No valid peptide features computed!")

    X, valid_indices = features_result
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    y = df_valid["label"].values

    print(f"  Valid features: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Class distribution: {np.bincount(y)}")

    return df_valid, X, y


def train_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """Train and evaluate the peptide model."""
    print("\n" + "=" * 60)
    print("TRAINING DPP4 PEPTIDE MODEL")
    print("=" * 60)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Remove zero-variance features (common in sparse DPC)
    variance = np.var(X_scaled, axis=0)
    kept_features = variance > 1e-10
    X_scaled = X_scaled[:, kept_features]
    print(f"\nFeatures after variance filter: {X_scaled.shape[1]} (removed {(~kept_features).sum()} zero-variance)")

    # Model: Gradient Boosting — good for small datasets with many features
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42,
    )

    # Cross-validation
    print("\n5-Fold Stratified Cross-Validation:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_auc = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
    cv_acc = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1")

    print(f"  ROC-AUC: {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")
    print(f"  Accuracy: {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")
    print(f"  F1-Score: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    # Train final model on all data
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    print(f"\nFull-dataset metrics (train):")
    print(f"  ROC-AUC: {roc_auc_score(y, y_prob):.3f}")
    print(f"  Accuracy: {accuracy_score(y, y_pred):.3f}")
    print(f"  Precision: {precision_score(y, y_pred):.3f}")
    print(f"  Recall: {recall_score(y, y_pred):.3f}")
    print(f"  F1: {f1_score(y, y_pred):.3f}")

    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=["Inactive", "Active"]))

    return model, scaler, kept_features


def validate_known_peptides(model, scaler, kept_features):
    """Validate model discriminates active from inactive peptides correctly."""
    print("\n" + "=" * 60)
    print("BIOLOGICAL BENCHMARK VALIDATION")
    print("=" * 60)

    # Known DPP4-inhibitory peptides (all have IC50 < 200 µM — should score HIGH)
    active_peptides = ["IPI", "IPP", "APLRW", "SPGPW", "FLQP", "YPFPGPIPN",
                       "GPAG", "GPRP", "VPGLAL", "APGPAGP", "GPP", "LKPNM"]
    # Known inactive peptides (IC50 > 500 µM — should score LOW)
    inactive_peptides = ["AA", "GG", "EE", "DD", "EEEE", "KKKK", "DDDD",
                         "SSSS", "NNNN", "RRRR"]

    print("\nActive DPP4-inhibitory peptides (should score > 50%):")
    active_scores = []
    for seq in active_peptides:
        fv = featurize_peptide(seq, "dpp4")
        if fv is not None:
            fv_scaled = scaler.transform(fv.reshape(1, -1))[:, kept_features]
            prob = model.predict_proba(fv_scaled)[0, 1]
            active_scores.append(prob)
            status = "✓" if prob > 0.5 else "✗"
            print(f"  {status} {seq:12s} → {prob*100:.1f}%")

    print("\nInactive peptides (should score < 50%):")
    inactive_scores = []
    for seq in inactive_peptides:
        fv = featurize_peptide(seq, "dpp4")
        if fv is not None:
            fv_scaled = scaler.transform(fv.reshape(1, -1))[:, kept_features]
            prob = model.predict_proba(fv_scaled)[0, 1]
            inactive_scores.append(prob)
            status = "✓" if prob < 0.5 else "✗"
            print(f"  {status} {seq:12s} → {prob*100:.1f}%")

    # Validation criteria
    mean_active = np.mean(active_scores) if active_scores else 0
    mean_inactive = np.mean(inactive_scores) if inactive_scores else 0
    separation = mean_active - mean_inactive

    print(f"\n{'─' * 40}")
    print(f"  Mean ACTIVE score:   {mean_active*100:.1f}%")
    print(f"  Mean INACTIVE score: {mean_inactive*100:.1f}%")
    print(f"  Separation:          {separation*100:.1f} percentage points")

    # Pass criteria: clear separation between active and inactive
    ranking_ok = (mean_active > 0.7 and mean_inactive < 0.3 and separation > 0.5)
    print(f"\n  Benchmark: {'✓ PASS' if ranking_ok else '✗ FAIL'}")
    print(f"    Active mean > 70%: {'✓' if mean_active > 0.7 else '✗'}")
    print(f"    Inactive mean < 30%: {'✓' if mean_inactive < 0.3 else '✗'}")
    print(f"    Separation > 50pp: {'✓' if separation > 0.5 else '✗'}")

    # Also test novel peptides NOT in training data
    print("\n\nNovel peptide predictions (NOT in training data):")
    novel = ["IPAV", "WPLG", "GPFP", "EKSS", "DENG", "IPPF"]
    for seq in novel:
        fv = featurize_peptide(seq, "dpp4")
        if fv is not None:
            fv_scaled = scaler.transform(fv.reshape(1, -1))[:, kept_features]
            prob = model.predict_proba(fv_scaled)[0, 1]
            print(f"  {seq:12s} → {prob*100:.1f}%")

    return ranking_ok


def save_model(model, scaler, kept_features, X):
    """Save trained model and artifacts."""
    print(f"\nSaving model to {MODEL_OUTPUT}")
    # Save model with metadata
    model_data = {
        "model": model,
        "scaler": scaler,
        "kept_features": kept_features,
        "model_type": "peptide_specific",
        "target": "dpp4",
        "feature_type": "sequence_descriptors",
        "n_features_input": 500,
        "n_features_used": int(kept_features.sum()),
        "activity_threshold_um": ACTIVITY_THRESHOLD_UM,
    }
    joblib.dump(model_data, MODEL_OUTPUT)

    # Save features for reference
    np.save(FEATURES_OUTPUT, X)
    print(f"Saved features to {FEATURES_OUTPUT}")
    print("Done!")


def main():
    df, X, y = load_and_prepare_data()
    model, scaler, kept_features = train_model(X, y)
    ranking_ok = validate_known_peptides(model, scaler, kept_features)

    if ranking_ok:
        save_model(model, scaler, kept_features, X)
        print("\n✓ Model passed biological benchmark — saved successfully.")
    else:
        print("\n✗ Model FAILED biological benchmark — NOT saved.")
        print("  Investigate data quality or feature engineering.")
        sys.exit(1)


if __name__ == "__main__":
    main()
