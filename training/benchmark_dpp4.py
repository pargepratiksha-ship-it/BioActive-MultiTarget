"""
DPP4 Biological Benchmark Validation.

Tests the trained DPP4 model against a curated panel of:
    - Known potent DPP4 inhibitors (must score highest)
    - Weak/moderate DPP4 inhibitors (should score intermediate)
    - Unrelated compounds with no known DPP4 activity (must score lowest)

A model passes only if the ranking order is preserved:
    potent > weak > unrelated

This is a REQUIRED gate before any model can be accepted.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Configuration
MODEL_PATH = Path("models/dpp4_model.joblib")
RESULTS_DIR = Path("results")
BENCHMARK_DIR = Path("data/benchmarks")
LOG_DIR = Path("logs")
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "dpp4_benchmark.log"),
    ],
)
logger = logging.getLogger(__name__)


# ── Benchmark Compound Panel ──────────────────────────────────────────────────
# Each compound has: name, SMILES, category, and literature IC50 where available.
# Sources: DrugBank, ChEMBL, published literature.

BENCHMARK_COMPOUNDS = [
    # ── Known potent DPP4 inhibitors (FDA-approved gliptins) ──
    {
        "name": "Sitagliptin",
        "smiles": "C(CC1=CC(=C(C=C1F)F)F)C(CC(=O)N1CCN2C(C1)CC(CC2)F)N1C=C(N=N1)C(F)(F)F",
        "category": "potent",
        "notes": "FDA-approved, IC50 ~18 nM",
    },
    {
        "name": "Vildagliptin",
        "smiles": "O=C(CN1CCC[C@@H]1C#N)[NH]C1CC2(O)CC(C1)C2",
        "category": "potent",
        "notes": "FDA-approved, IC50 ~3.5 nM",
    },
    {
        "name": "Saxagliptin",
        "smiles": "O=C(C(N)C12CC3CC(C1)CC(O)(C3)C2)N1[C@H](C#N)C[C@@H]2C[C@H]21",
        "category": "potent",
        "notes": "FDA-approved, IC50 ~1.3 nM",
    },
    {
        "name": "Linagliptin",
        "smiles": "CC#CC1=NN2C(=N1)N(C(=O)C2=O)CC1=CC=C(C=C1)NC1=NC(=NC=C1)N1CCN(CC1)C",
        "category": "potent",
        "notes": "FDA-approved, IC50 ~1 nM",
    },
    {
        "name": "Alogliptin",
        "smiles": "O=C1N(CC2=CC=C(C=C2)CN2C=NC3=C2C(=O)N(C(=O)N3)C)CC(=O)N1",
        "category": "potent",
        "notes": "FDA-approved, IC50 ~6.9 nM",
    },
    # ── Weak/moderate DPP4 inhibitors ──
    {
        "name": "Diprotin A (Ile-Pro-Ile)",
        "smiles": "CC(C)C(N)C(=O)N1CCCC1C(=O)NC(CC(C)C)C(=O)O",
        "category": "weak",
        "notes": "Tripeptide DPP4 inhibitor, IC50 ~4.2 µM",
    },
    {
        "name": "Vildagliptin carboxylic acid metabolite",
        "smiles": "OC(=O)CN1CCCC1C(=O)NC1CC2(O)CC(C1)C2",
        "category": "weak",
        "notes": "Primary metabolite, ~100x less potent than vildagliptin",
    },
    # ── Unrelated compounds (no DPP4 activity) ──
    {
        "name": "Aspirin",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "category": "unrelated",
        "notes": "NSAID, no known DPP4 activity",
    },
    {
        "name": "Amoxicillin",
        "smiles": "CC1(C)SC2C(NC(=O)C(N)C3=CC=C(O)C=C3)C(=O)N2C1C(=O)O",
        "category": "unrelated",
        "notes": "Antibiotic, no known DPP4 activity",
    },
    {
        "name": "Metformin",
        "smiles": "CN(C)C(=N)NC(=N)N",
        "category": "unrelated",
        "notes": "Antidiabetic (AMPK pathway), no direct DPP4 inhibition",
    },
    {
        "name": "Ibuprofen",
        "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "category": "unrelated",
        "notes": "NSAID, no known DPP4 activity",
    },
    {
        "name": "Caffeine",
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "category": "unrelated",
        "notes": "Stimulant, no known DPP4 activity",
    },
]


def smiles_to_fingerprint(smiles: str) -> np.ndarray | None:
    """Convert SMILES to Morgan fingerprint.

    Args:
        smiles: SMILES string.

    Returns:
        Fingerprint array or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
    return np.array(fp)


def run_benchmark() -> dict:
    """Score all benchmark compounds and evaluate ranking.

    Returns:
        Dictionary with benchmark results and pass/fail status.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DPP4 Biological Benchmark Validation")
    logger.info("=" * 60)

    # Load model
    if not MODEL_PATH.exists():
        logger.error(f"Model not found at {MODEL_PATH}. Run train_dpp4.py first.")
        return {"passed": False, "error": "Model not found"}

    model = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")

    # Score each compound
    results = []
    for compound in BENCHMARK_COMPOUNDS:
        fp = smiles_to_fingerprint(compound["smiles"])
        if fp is None:
            logger.warning(f"Invalid SMILES for {compound['name']}, skipping")
            continue

        prob = model.predict_proba(fp.reshape(1, -1))[0, 1]
        results.append({
            "name": compound["name"],
            "category": compound["category"],
            "dpp4_probability": float(prob),
            "notes": compound["notes"],
        })
        logger.info(f"  {compound['name']:30s} | {compound['category']:10s} | P(active)={prob:.4f}")

    results_df = pd.DataFrame(results)

    # Compute category-level statistics
    category_stats = {}
    for category in ["potent", "weak", "unrelated"]:
        cat_probs = results_df[results_df["category"] == category]["dpp4_probability"]
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

    # Ranking test: potent_mean > weak_mean > unrelated_mean
    potent_mean = category_stats.get("potent", {}).get("mean", 0)
    weak_mean = category_stats.get("weak", {}).get("mean", 0)
    unrelated_mean = category_stats.get("unrelated", {}).get("mean", 0)

    rank_potent_vs_weak = potent_mean > weak_mean
    rank_weak_vs_unrelated = weak_mean > unrelated_mean
    rank_potent_vs_unrelated = potent_mean > unrelated_mean

    # Strict test: minimum potent score > maximum unrelated score
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
    logger.info(f"\n  OVERALL: {'PASS ✓' if passed else 'FAIL ✗'}")
    logger.info("=" * 60)

    # Save results
    benchmark_results = {
        "target": "DPP4",
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

    results_path = RESULTS_DIR / "dpp4_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Save benchmark panel as CSV for reference
    results_df.to_csv(BENCHMARK_DIR / "dpp4_benchmark_panel.csv", index=False)

    return benchmark_results


if __name__ == "__main__":
    run_benchmark()
