"""Tests for DPP4 benchmark validation."""

import numpy as np
import pytest
from rdkit import Chem

from training.benchmark_dpp4 import BENCHMARK_COMPOUNDS, smiles_to_fingerprint


class TestBenchmarkCompounds:
    """Validate the benchmark compound panel itself."""

    def test_all_smiles_valid(self):
        """Every benchmark compound must have a valid, parseable SMILES."""
        for compound in BENCHMARK_COMPOUNDS:
            mol = Chem.MolFromSmiles(compound["smiles"])
            assert mol is not None, f"Invalid SMILES for {compound['name']}: {compound['smiles']}"

    def test_categories_present(self):
        """Panel must include potent, weak, and unrelated categories."""
        categories = {c["category"] for c in BENCHMARK_COMPOUNDS}
        assert "potent" in categories
        assert "weak" in categories
        assert "unrelated" in categories

    def test_minimum_potent_inhibitors(self):
        """Must have at least 3 known potent inhibitors for statistical robustness."""
        potent = [c for c in BENCHMARK_COMPOUNDS if c["category"] == "potent"]
        assert len(potent) >= 3

    def test_minimum_unrelated_compounds(self):
        """Must have at least 3 unrelated compounds."""
        unrelated = [c for c in BENCHMARK_COMPOUNDS if c["category"] == "unrelated"]
        assert len(unrelated) >= 3

    def test_no_duplicate_names(self):
        names = [c["name"] for c in BENCHMARK_COMPOUNDS]
        assert len(names) == len(set(names))


class TestFingerprinting:
    """Tests for benchmark fingerprint generation."""

    def test_benchmark_fingerprints(self):
        """All benchmark compounds must produce valid fingerprints."""
        for compound in BENCHMARK_COMPOUNDS:
            fp = smiles_to_fingerprint(compound["smiles"])
            assert fp is not None, f"Fingerprint failed for {compound['name']}"
            assert fp.shape == (2048,)
