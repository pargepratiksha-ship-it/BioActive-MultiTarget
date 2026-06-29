"""
Setup script — trains all models from raw data.

Run this once on a fresh machine to generate all .joblib model files
and processed data needed by the Streamlit app.

Usage:
    python setup_models.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def run_script(script_path: str, description: str) -> bool:
    """Run a Python script and report success/failure."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    full_path = PROJECT_ROOT / script_path
    if not full_path.exists():
        print(f"  SKIP: {script_path} not found")
        return False
    result = subprocess.run(
        [sys.executable, str(full_path)],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        print(f"  ✓ {description} — SUCCESS")
        return True
    else:
        print(f"  ✗ {description} — FAILED (exit code {result.returncode})")
        return False


def main():
    print("BioActive-MultiTarget — Model Setup")
    print("This will train all models from raw/processed data.\n")

    # Ensure output directories exist
    (PROJECT_ROOT / "models").mkdir(exist_ok=True)
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    (PROJECT_ROOT / "results").mkdir(exist_ok=True)

    results = {}

    # Train all small-molecule models
    targets = ["dpp4", "amylase", "glucosidase", "lipase", "ace",
               "egfr", "her2", "vegfr2", "braf", "cdk2"]

    for target in targets:
        script = f"training/train_{target}.py"
        ok = run_script(script, f"Training {target.upper()} model")
        results[target] = ok

    # Train peptide model(s)
    ok = run_script("training/train_dpp4_peptide.py", "Training DPP4 peptide model")
    results["dpp4_peptide"] = ok

    # Summary
    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED/SKIPPED"
        print(f"  {name:20s} {status}")

    n_ok = sum(results.values())
    n_total = len(results)
    print(f"\n  {n_ok}/{n_total} models ready.")

    if n_ok > 0:
        print(f"\n  You can now run the app:")
        print(f"    streamlit run app/main.py")
    else:
        print(f"\n  No models were trained. Check that data files exist in data/raw/")


if __name__ == "__main__":
    main()
