# BioActive-MultiTarget — Project Guidelines

## Critical Rules

1. **Biological realism and scientific validity are the highest priority.** Never optimize solely for accuracy, ROC-AUC, F1, or dataset size.
2. **Never build a universal bioactivity classifier.** Every model is target-specific.
3. **Never combine unrelated biological targets into one model.**
4. **Unknown target-ligand pairs are NOT true negatives.** Use explicit labeling strategies.
5. **Every model must pass biological benchmark tests before acceptance.** ML metrics alone are insufficient.
6. **Small-molecule models and peptide models remain separate** unless sufficient peptide-labelled data exists for that target.
7. **Do not claim peptide prediction capability without peptide-labelled training data.**
8. **Biological plausibility > benchmark metrics.** A model that ranks known inhibitors above unrelated compounds is more important than one with 0.99 AUC on a flawed test set.

---

## Architecture

### Platform Purpose

Target-specific bioactivity screening for small molecules and bioactive peptides, supporting discovery of bioactive peptides from marine fish collagen, porcine gelatin, food proteins, enzymatic hydrolysates, and experimentally identified peptides.

### Workflow

```
Protein Source → In-silico Hydrolysis → Peptide Generation → BioActive-MultiTarget → Target-Specific Prediction → Candidate Ranking → Wet-Lab Validation
```

### Project Structure

```
BioActive_MultiTarget/
├── data/
│   ├── raw/              # Unprocessed ChEMBL extractions
│   ├── processed/        # Cleaned, featurized datasets
│   └── benchmarks/       # Known inhibitor panels for validation
├── models/               # Serialized trained models (target_model.joblib)
├── sequences/            # Protein/peptide sequence data
├── training/             # Training scripts and configs
├── app/                  # Streamlit web application
├── results/              # Evaluation outputs, plots, reports
├── logs/                 # Training and pipeline logs
└── docs/                 # Documentation, roadmaps
```

### Independent Models (one per target, never merged)

Phase 1 — Diabetes: `dpp4_model`, `amylase_model`, `glucosidase_model`, `lipase_model`
Phase 2 — Hypertension: `ace_model`
Phase 3 — Cancer: `egfr_model`, `her2_model`, `vegfr2_model`, `braf_model`, `cdk2_model`

---

## Tech Stack

- **Language:** Python 3.10+
- **Web Framework:** Streamlit
- **ML:** scikit-learn, XGBoost, optionally PyTorch (GNN)
- **Cheminformatics:** RDKit (Morgan fingerprints, molecule handling)
- **Protein Embeddings:** ESM2 (via transformers/fair-esm)
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **API Access:** chembl_webresource_client (ChEMBL 37)
- **Serialization:** joblib
- **Testing:** pytest

---

## Build and Test

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run the web app
streamlit run app/main.py
```

---

## Conventions

### Naming

- Model files: `{target}_model.joblib` (e.g., `dpp4_model.joblib`)
- Dataset files: `{target}_{stage}.csv` (e.g., `dpp4_raw.csv`, `dpp4_processed.csv`)
- Training scripts: `train_{target}.py`
- Feature scripts: `featurize_{target}.py`

### Code Style

- PEP 8 with 120 char line length
- Type hints on all public function signatures
- Docstrings (Google style) for modules, classes, and public functions
- Constants in UPPER_SNAKE_CASE

### Data Handling

- Raw data is never modified in place — always write to `processed/`
- Every dataset must track: source, version, extraction date, positive/negative counts
- Duplicate removal by canonical SMILES (small molecules) or normalized sequence (peptides)
- Activity threshold must be explicit and documented per target (e.g., IC50 ≤ 10 µM = active)

### Model Validation Requirements

Every model must report: ROC-AUC, Accuracy, Precision, Recall, F1, positive count, negative count.

Every model must pass **biological benchmark ranking**:
- Known potent inhibitors must score higher than unrelated compounds
- Weak inhibitors should score intermediate
- Decoy/unrelated compounds must score lowest

### Scientific Guardrails

- Do NOT proceed to the next target until the current target passes benchmark validation
- Do NOT mix ChEMBL targets (e.g., human vs. bacterial amylase) without explicit justification
- Do NOT use random compounds as "negatives" without verifying they are truly inactive against the target
- Document all activity cutoffs, assay types, and confidence scores used in filtering

---

## Data Sources

| Source | Use |
|--------|-----|
| ChEMBL 37 | Primary small-molecule bioactivity data |
| BIOPEP-UWM | Future peptide bioactivity (DPP4, ACE inhibitory peptides) |
| SATPdb | Future peptide data |
| AHTPDB | Antihypertensive peptides |
| CancerPPD | Anticancer peptides |
| Published literature | Curated peptide inhibition studies |

---

## Current Priority

**DPP4 (CHEMBL284) only.** Do not begin other targets until DPP4 is scientifically validated.

---

## Development Roadmap

### Phase 0 — Project Setup & Scientific Review

- [ ] 0.1 Scaffold project structure (directories, `requirements.txt`, README)
- [ ] 0.2 Critical scientific review: identify risks, data limitations, realistic capabilities
- [ ] 0.3 Document what can vs. cannot be predicted with available data
- [ ] 0.4 Define peptide data requirements for future phases

### Phase 1A — DPP4 Data Pipeline

- [ ] 1.1 Extract DPP4 (CHEMBL284) bioactivity data from ChEMBL 37
- [ ] 1.2 Filter by assay type, confidence score, activity type (IC50/Ki)
- [ ] 1.3 Define and document activity threshold (e.g., IC50 ≤ 10 µM = active)
- [ ] 1.4 Remove duplicates (canonical SMILES deduplication)
- [ ] 1.5 Handle conflicting measurements (same compound, different results)
- [ ] 1.6 Split into train/validation/test sets (scaffold split preferred)
- [ ] 1.7 Version the dataset with metadata (date, counts, filters applied)

### Phase 1B — DPP4 Feature Engineering

- [ ] 1.8 Compute Morgan fingerprints (radius=2, 2048 bits) for all compounds
- [ ] 1.9 Validate fingerprint generation (no NaN, correct dimensionality)
- [ ] 1.10 Optional: compute additional descriptors (MW, LogP, TPSA)

### Phase 1C — DPP4 Model Training

- [ ] 1.11 Train baseline model (Random Forest or XGBoost)
- [ ] 1.12 Cross-validation (5-fold stratified)
- [ ] 1.13 Hyperparameter tuning (Optuna or GridSearch)
- [ ] 1.14 Report all metrics (AUC, Acc, Prec, Recall, F1, sample counts)
- [ ] 1.15 Save model as `dpp4_model.joblib`

### Phase 1D — DPP4 Biological Benchmark Validation

- [ ] 1.16 Curate benchmark panel:
  - Known potent DPP4 inhibitors (sitagliptin, vildagliptin, saxagliptin, linagliptin, alogliptin)
  - Weak/moderate DPP4 inhibitors
  - Unrelated compounds (e.g., antibiotics, NSAIDs with no DPP4 activity)
- [ ] 1.17 Score all benchmark compounds with the trained model
- [ ] 1.18 Verify ranking: potent > weak > unrelated
- [ ] 1.19 If ranking fails → diagnose, retrain, adjust thresholds
- [ ] 1.20 Document benchmark results and acceptance decision

### Phase 1E — DPP4 Streamlit App

- [ ] 1.21 Build Streamlit UI: SMILES input → DPP4 inhibition probability
- [ ] 1.22 Display molecule structure (rdkit depiction)
- [ ] 1.23 Show prediction confidence and applicability domain warning
- [ ] 1.24 Batch prediction mode (CSV upload)

### Phase 2 — Expand to Remaining Diabetes Targets

- [ ] 2.1 Alpha-Amylase pipeline (repeat 1.1–1.20)
- [ ] 2.2 Alpha-Glucosidase pipeline
- [ ] 2.3 Pancreatic Lipase pipeline
- [ ] 2.4 Integrate all Phase 1 models into unified Streamlit app

### Phase 3 — ACE (Hypertension)

- [ ] 3.1 ACE model pipeline
- [ ] 3.2 Benchmark with known ACE inhibitors (captopril, enalaprilat, lisinopril)

### Phase 4 — Cancer Targets

- [ ] 4.1–4.5 Independent models for EGFR, HER2, VEGFR2, BRAF, CDK2

### Phase 5 — Peptide Screening

- [ ] 5.1 Acquire peptide-labelled datasets (BIOPEP, AHTPDB, literature)
- [ ] 5.2 Build peptide featurization (sequence embeddings, physicochemical descriptors)
- [ ] 5.3 Train target-specific peptide models (only for targets with sufficient peptide data)
- [ ] 5.4 In-silico hydrolysis module (PeptideCutter-style digestion)
- [ ] 5.5 Full peptide screening workflow in Streamlit

### Phase 6 — Production Hardening

- [ ] 6.1 Applicability domain checks for all models
- [ ] 6.2 Confidence calibration
- [ ] 6.3 Logging, error handling, deployment packaging

---

## Known Risks & Limitations

1. **Peptide prediction requires peptide-labelled data.** Small-molecule models trained on ChEMBL cannot reliably predict peptide bioactivity. Peptide screening (Phase 5) depends on acquiring suitable training data.
2. **Activity cliffs:** Structurally similar compounds can have vastly different activities. Morgan fingerprints may not capture all relevant SAR.
3. **Assay heterogeneity:** ChEMBL aggregates data from many labs/assays. Filter by confidence score ≥ 8 and standardized activity types.
4. **Class imbalance:** Many targets will have far more inactives than actives. Use stratified splitting and appropriate metrics.
5. **Applicability domain:** Predictions outside the chemical space of training data are unreliable. Must warn users.
6. **Negative labeling problem:** Compounds not tested against a target are NOT confirmed negatives. Use only explicitly tested compounds.

---

## What Can and Cannot Be Predicted

### CAN predict (with current ChEMBL data):
- Small-molecule inhibition probability for specific targets
- Relative ranking of small molecules by predicted bioactivity
- Chemical similarity to known inhibitors

### CANNOT predict (without peptide training data):
- Peptide bioactivity from small-molecule models alone
- Mechanism of action
- In-vivo efficacy from in-vitro activity predictions
- Selectivity across targets (requires multi-target comparison, not a single model)

### FUTURE capability (with peptide datasets):
- Target-specific peptide inhibition probability
- Peptide candidate ranking from hydrolysate libraries
