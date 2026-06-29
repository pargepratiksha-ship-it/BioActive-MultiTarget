"""
Generic Pipeline Runner — Run the full pipeline for any target.

Usage:
    python training/run_target_pipeline.py amylase
    python training/run_target_pipeline.py ace
    python training/run_target_pipeline.py egfr
    python training/run_target_pipeline.py --all
    python training/run_target_pipeline.py --list

Supported targets: amylase, glucosidase, lipase, ace, egfr, her2, vegfr2, braf, cdk2
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.pipeline import run_full_pipeline
from training.target_configs import ALL_TARGETS, get_config


def main():
    """Parse arguments and run pipeline for specified target(s)."""
    if len(sys.argv) < 2:
        print("Usage: python training/run_target_pipeline.py <target_name|--all|--list>")
        print(f"Available targets: {', '.join(sorted(ALL_TARGETS.keys()))}")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--list":
        print("Available targets:")
        for name, config in sorted(ALL_TARGETS.items()):
            print(f"  {name:15s} | {config['chembl_id']:12s} | {config['description']}")
        return

    if arg == "--all":
        targets = list(ALL_TARGETS.keys())
    else:
        targets = [arg]

    failed = []
    passed = []

    for target_name in targets:
        try:
            config = get_config(target_name)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

        print(f"\n{'#' * 70}")
        print(f"# Target: {config['display_name']} ({config['chembl_id']})")
        print(f"# Threshold: {config['activity_threshold_nM']} nM")
        print(f"{'#' * 70}\n")

        results = run_full_pipeline(config)

        if results.get("passed"):
            passed.append(target_name)
        else:
            failed.append(target_name)

    # Summary
    if len(targets) > 1:
        print("\n" + "=" * 70)
        print("MULTI-TARGET PIPELINE SUMMARY")
        print("=" * 70)
        print(f"  Passed: {len(passed)} — {', '.join(passed) if passed else 'none'}")
        print(f"  Failed: {len(failed)} — {', '.join(failed) if failed else 'none'}")
        print("=" * 70)


if __name__ == "__main__":
    main()
