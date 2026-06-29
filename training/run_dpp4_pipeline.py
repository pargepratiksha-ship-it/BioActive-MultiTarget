"""
DPP4 Full Pipeline Runner.

Runs the complete DPP4 pipeline in order:
    1. Extract data from ChEMBL
    2. Process and clean data
    3. Featurize compounds
    4. Train model
    5. Run biological benchmark validation

Usage:
    python training/run_dpp4_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Execute the full DPP4 pipeline."""
    print("=" * 60)
    print("DPP4 FULL PIPELINE")
    print("=" * 60)

    # Step 1: Extract
    print("\n[1/5] Extracting DPP4 data from ChEMBL...")
    from training.extract_dpp4 import main as extract_main
    extract_main()

    # Step 2: Process
    print("\n[2/5] Processing and cleaning data...")
    from training.process_dpp4 import process_dpp4_data
    process_dpp4_data()

    # Step 3: Featurize
    print("\n[3/5] Computing molecular fingerprints...")
    from training.featurize_dpp4 import featurize_dpp4
    featurize_dpp4()

    # Step 4: Train
    print("\n[4/5] Training DPP4 model...")
    from training.train_dpp4 import train_dpp4_model
    train_dpp4_model()

    # Step 5: Benchmark
    print("\n[5/5] Running biological benchmark validation...")
    from training.benchmark_dpp4 import run_benchmark
    results = run_benchmark()

    print("\n" + "=" * 60)
    if results.get("passed"):
        print("PIPELINE COMPLETE — DPP4 model PASSED benchmark validation")
    else:
        print("PIPELINE COMPLETE — DPP4 model FAILED benchmark validation")
        print("Review results/dpp4_benchmark_results.json for details.")
    print("=" * 60)


if __name__ == "__main__":
    main()
