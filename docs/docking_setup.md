# Molecular Docking Setup — AutoDock Vina

## Overview

The docking pipeline uses AutoDock Vina via command-line interface for peptide scoring.
This provides structure-based predictions complementing the physicochemical and ML scores.

## Installation

### Option 1: Download Vina Binary (Recommended)

1. Download AutoDock Vina from: https://vina.scripps.edu/downloads/
2. Extract and add the binary to your PATH
3. Verify: `vina --version`

### Option 2: Conda (if available)

```bash
conda install -c conda-forge autodock-vina
```

## Receptor Preparation

Each target requires a prepared receptor PDBQT file. Place them in `receptors/`:

```
receptors/
├── 1X70.pdbqt    # ACE
├── 2ONC.pdbqt    # DPP4
├── 1SMD.pdbqt    # Alpha-Amylase
├── 3TOP.pdbqt    # Alpha-Glucosidase
├── 1LPB.pdbqt    # Pancreatic Lipase
├── 1M17.pdbqt    # EGFR
├── 3PP0.pdbqt    # HER2
├── 2OH4.pdbqt    # VEGFR2
├── 3OG7.pdbqt    # BRAF
├── 1DI8.pdbqt    # CDK2
```

### How to prepare a receptor PDBQT:

1. Download PDB structure from RCSB PDB
2. Remove water molecules, cofactors, and existing ligands
3. Add hydrogens (at physiological pH)
4. Convert to PDBQT using `prepare_receptor4.py` from ADFRsuite:

```bash
# Install ADFRsuite
# https://ccsb.scripps.edu/adfr/downloads/

# Prepare receptor
prepare_receptor4.py -r protein.pdb -o protein.pdbqt
```

Or use Open Babel:
```bash
obabel protein.pdb -O protein.pdbqt -xr
```

## Docking Box Configuration

The docking box for each target is pre-configured in `training/docking_pipeline.py`.
These coordinates center on the active site based on co-crystallized ligand positions.

## Dependencies

- **meeko** (installed): Ligand PDBQT preparation from SMILES
- **biopython** (installed): PDB file handling
- **AutoDock Vina binary**: Must be on PATH

## Usage

```python
from training.docking_pipeline import run_docking, compute_composite_score
from training.peptide_scoring import compute_compatibility_score

# Docking score (requires Vina + receptor files)
docking_result = run_docking("SMILES_STRING", "ace")

# Physicochemical score (always available)
compat_result = compute_compatibility_score("IPP", "ace")

# Composite (when docking available)
composite = compute_composite_score("IPP", "ace", docking_score=-8.5)
```

## Status

- ✅ Physicochemical scoring: Working (no external dependencies)
- ✅ ACE peptide ML model: Working (trained on literature data)
- ⏳ Docking pipeline: Code complete, awaiting Vina binary + receptor files
