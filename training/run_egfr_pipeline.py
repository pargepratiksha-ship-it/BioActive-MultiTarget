"""EGFR (Epidermal Growth Factor Receptor) Full Pipeline Runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.pipeline import run_full_pipeline
from training.target_configs import EGFR_CONFIG

if __name__ == "__main__":
    run_full_pipeline(EGFR_CONFIG)
