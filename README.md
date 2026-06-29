# BioActive-MultiTarget

Target-specific bioactivity screening platform for small molecules and bioactive peptides.

## Purpose

Predict bioactivity of compounds against specific therapeutic targets to support discovery of bioactive peptides from food proteins, enzymatic hydrolysates, and marine collagen sources.

## Current Status

**Phase 1A — DPP4 Pipeline** (in progress)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
streamlit run app/main.py

# Run tests
pytest tests/ -v
```

## Project Structure

```
data/raw/           — Unprocessed ChEMBL extractions
data/processed/     — Cleaned, featurized datasets
data/benchmarks/    — Known inhibitor panels for validation
models/             — Serialized trained models
sequences/          — Protein/peptide sequence data
training/           — Training scripts and configs
app/                — Streamlit web application
results/            — Evaluation outputs, plots, reports
tests/              — Test suite
```

## Targets

| Phase | Target | ChEMBL ID | Status |
|-------|--------|-----------|--------|
| 1 | DPP4 | CHEMBL284 | In progress |
| 1 | Alpha-Amylase | TBD | Pending |
| 1 | Alpha-Glucosidase | TBD | Pending |
| 1 | Pancreatic Lipase | TBD | Pending |
| 2 | ACE | TBD | Pending |
| 3 | EGFR, HER2, VEGFR2, BRAF, CDK2 | — | Pending |

## Scientific Principles

- One model per target — never merged
- Biological benchmark validation required before acceptance
- Small-molecule and peptide models remain separate
- Unknown pairs ≠ true negatives
