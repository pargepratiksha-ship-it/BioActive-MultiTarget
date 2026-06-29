"""HER2 (Human Epidermal Growth Factor Receptor 2) Full Pipeline Runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.pipeline import run_full_pipeline
from training.target_configs import HER2_CONFIG

if __name__ == "__main__":
    run_full_pipeline(HER2_CONFIG)
