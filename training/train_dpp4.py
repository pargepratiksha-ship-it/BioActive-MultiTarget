"""
DPP4 Model Training.

Trains an XGBoost classifier on Morgan fingerprints for DPP4 bioactivity prediction.
Includes cross-validation, hyperparameter tuning via Optuna, and full metric reporting.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Configuration
PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
LOG_DIR = Path("logs")
N_CV_FOLDS = 5
N_OPTUNA_TRIALS = 50
RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "dpp4_training.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_features(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed fingerprints and labels.

    Args:
        split: 'train' or 'test'.

    Returns:
        Tuple of (X, y) arrays.
    """
    X = np.load(PROCESSED_DIR / f"dpp4_{split}_fingerprints.npy")
    y = np.load(PROCESSED_DIR / f"dpp4_{split}_labels.npy")
    return X, y


def evaluate_model(model: XGBClassifier, X: np.ndarray, y: np.ndarray, label: str) -> dict:
    """Compute all required metrics.

    Args:
        model: Trained classifier.
        X: Feature matrix.
        y: True labels.
        label: Description for logging (e.g., 'train', 'test').

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


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    """Optuna objective for XGBoost hyperparameter tuning.

    Args:
        trial: Optuna trial object.
        X: Training feature matrix.
        y: Training labels.

    Returns:
        Mean cross-validation ROC-AUC.
    """
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

    # Calculate scale_pos_weight for class imbalance
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


def train_dpp4_model() -> None:
    """Run the full DPP4 model training pipeline."""
    for d in [MODEL_DIR, RESULTS_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DPP4 Model Training Pipeline")
    logger.info("=" * 60)

    # Load data
    X_train, y_train = load_features("train")
    X_test, y_test = load_features("test")
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Train labels: {int(y_train.sum())} active, {int(len(y_train) - y_train.sum())} inactive")
    logger.info(f"Test labels: {int(y_test.sum())} active, {int(len(y_test) - y_test.sum())} inactive")

    # Hyperparameter tuning with Optuna
    logger.info(f"Starting Optuna hyperparameter search ({N_OPTUNA_TRIALS} trials)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=N_OPTUNA_TRIALS)

    best_params = study.best_params
    logger.info(f"Best CV ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {best_params}")

    # Train final model with best params on full training set
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
    model_path = MODEL_DIR / "dpp4_model.joblib"
    joblib.dump(final_model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save results
    results = {
        "target": "DPP4",
        "training_date": datetime.now().isoformat(),
        "model_type": "XGBClassifier",
        "best_params": best_params,
        "best_cv_roc_auc": study.best_value,
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "n_cv_folds": N_CV_FOLDS,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }
    results_path = RESULTS_DIR / "dpp4_training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    train_dpp4_model()
