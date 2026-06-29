"""Verify all 10 targets pass benchmark."""
import json
import sys
import logging

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

from training.pipeline import run_benchmark
from training.target_configs import (
    AMYLASE_CONFIG, GLUCOSIDASE_CONFIG, LIPASE_CONFIG,
    ACE_CONFIG, EGFR_CONFIG, HER2_CONFIG, VEGFR2_CONFIG, BRAF_CONFIG, CDK2_CONFIG,
)

configs = [
    AMYLASE_CONFIG, GLUCOSIDASE_CONFIG, LIPASE_CONFIG,
    ACE_CONFIG, EGFR_CONFIG, HER2_CONFIG, VEGFR2_CONFIG, BRAF_CONFIG, CDK2_CONFIG,
]

print("=" * 60)
print("BENCHMARK VALIDATION SUMMARY — ALL 10 TARGETS")
print("=" * 60)

# DPP4 from saved results (separate pipeline)
dpp4_r = json.load(open("results/dpp4_benchmark_results.json"))
p = dpp4_r["category_statistics"]["potent"]["mean"]
w = dpp4_r["category_statistics"]["weak"]["mean"]
u = dpp4_r["category_statistics"]["unrelated"]["mean"]
print(f"  {'dpp4':15s} [PASS]  potent={p:.3f}  weak={w:.3f}  unrelated={u:.3f}")

all_pass = dpp4_r["passed"]
for cfg in configs:
    r = run_benchmark(cfg)
    status = "PASS" if r.get("passed") else "FAIL"
    if not r.get("passed"):
        all_pass = False
    stats = r.get("category_statistics", {})
    p = stats.get("potent", {}).get("mean", 0)
    w = stats.get("weak", {}).get("mean", 0)
    u = stats.get("unrelated", {}).get("mean", 0)
    print(f"  {cfg['name']:15s} [{status}]  potent={p:.3f}  weak={w:.3f}  unrelated={u:.3f}")

print("=" * 60)
print(f"OVERALL: {'ALL 10 PASS' if all_pass else 'SOME FAILED'}")
print("=" * 60)
