# BioActive-MultiTarget

Target-specific bioactivity screening platform for small molecules and bioactive peptides, supporting discovery from food proteins, enzymatic hydrolysates, and marine collagen sources.

## Current Status

All 10 small-molecule targets are trained, benchmark-validated, and available in the Streamlit app. DPP4 peptide model is also operational. Molecular docking (AutoDock Vina) is integrated for structure-based scoring when the binary is available.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train all models (required after fresh clone — models are gitignored)
python setup_models.py

# Verify all benchmarks pass
python verify_all_benchmarks.py

# Run the web app
streamlit run app/main.py

# Run tests
pytest tests/ -v
```

> **Note:** Model files (`models/*.joblib`), results, and processed data are excluded from version control. Run `setup_models.py` once to regenerate them.

## Prediction Pipeline

Every prediction follows a multi-stage workflow:

```
Input → Validation → ML Prediction (Morgan FP → XGBoost) →
Physicochemical Analysis (peptides) → Docking Feasibility →
[Molecular Docking] → Evidence Integration → Confidence & Status
```

Each prediction carries an explicit `EvaluationStatus`:
`FULLY_EVALUATED` | `PARTIALLY_EVALUATED` | `INVALID_INPUT` | `FAILED_PROCESSING`

## Project Structure

```
app/                — Streamlit web application
training/           — Training scripts, pipelines, and target configs
data/
  raw/              — Unprocessed ChEMBL extractions (gitignored CSVs)
  processed/        — Cleaned, featurized datasets (gitignored CSVs)
  benchmarks/       — Known inhibitor panels (gitignored CSVs)
  peptides/         — Peptide datasets (e.g., dpp4_peptides.csv)
models/             — Serialized trained models (gitignored .joblib)
receptors/          — Prepared receptor PDBQT files for docking
results/            — Benchmark/training results (gitignored)
tools/              — External binaries (e.g., AutoDock Vina)
docs/               — Documentation (docking setup, etc.)
sequences/          — Protein/peptide sequence data
tests/              — Test suite
scripts/            — Utility scripts
logs/               — Runtime logs (gitignored)
```

## Targets

| Phase | Target | ChEMBL ID | Model | Status |
|-------|--------|-----------|-------|--------|
| 1 | DPP4 | CHEMBL284 | `dpp4_model.joblib` | Trained & validated |
| 1 | Alpha-Amylase | CHEMBL2045 | `amylase_model.joblib` | Trained & validated |
| 1 | Alpha-Glucosidase | CHEMBL3833502 | `glucosidase_model.joblib` | Trained & validated |
| 1 | Pancreatic Lipase | CHEMBL1812 | `lipase_model.joblib` | Trained & validated |
| 2 | ACE | CHEMBL1808 | `ace_model.joblib` | Trained & validated |
| 3 | EGFR | CHEMBL203 | `egfr_model.joblib` | Trained & validated |
| 3 | HER2 | CHEMBL1824 | `her2_model.joblib` | Trained & validated |
| 3 | VEGFR2 | CHEMBL279 | `vegfr2_model.joblib` | Trained & validated |
| 3 | BRAF | CHEMBL5145 | `braf_model.joblib` | Trained & validated |
| 3 | CDK2 | CHEMBL301 | `cdk2_model.joblib` | Trained & validated |
| — | DPP4 (peptides) | CHEMBL284 | `dpp4_peptide_model.joblib` | Trained & validated |

## Molecular Docking

Optional structure-based scoring via AutoDock Vina. Receptor PDBQT files are checked into `receptors/`. See [docs/docking_setup.md](docs/docking_setup.md) for installation instructions.

Docking is automatically attempted when:
- The Vina binary is available (`tools/vina_1.2.5_win.exe` or on PATH)
- The docking feasibility check passes for the given ligand

## Peptide Scoring

Peptide predictions combine:
1. **ML model** (Morgan fingerprints, where peptide-labelled data exists)
2. **Physicochemical compatibility** (target-specific profiles from literature)
3. **Molecular docking** (when feasible and available)

DPP4 has a dedicated peptide ML model while the other targets score peptides via physicochemical compatibility and docking with small-molecule(small ligands except peptides) ML fallback (lower confidence).

## Scientific Principles

- One model per target — never merged
- Biological benchmark validation required before acceptance (potent > weak > unrelated)
- Small-molecule and peptide models remain separate
- Unknown pairs ≠ true negatives
- Applicability domain warnings for out-of-distribution inputs

## Key Scripts

| Script | Purpose |
|--------|---------|
| `setup_models.py` | Train all models from raw data |
| `verify_all_benchmarks.py` | Validate all 10 targets pass biological benchmarks |
| `training/run_{target}_pipeline.py` | Train a single target model |
| `training/pipeline.py` | Generic training/benchmark pipeline |
| `training/prediction_pipeline.py` | Multi-stage prediction orchestration |
| `training/docking_pipeline.py` | AutoDock Vina wrapper |
| `training/peptide_scoring.py` | Physicochemical peptide compatibility |
| `training/target_configs.py` | Target definitions and benchmark panels |