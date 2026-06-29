"""Tests for DPP4 feature engineering."""

import numpy as np
import pytest

from training.featurize_dpp4 import (
    MORGAN_NBITS,
    compute_descriptors,
    compute_morgan_fingerprint,
)


class TestMorganFingerprint:
    """Tests for Morgan fingerprint computation."""

    def test_valid_smiles(self):
        fp = compute_morgan_fingerprint("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert fp is not None
        assert fp.shape == (MORGAN_NBITS,)
        assert fp.dtype in [np.int8, np.int32, np.int64, np.uint8]

    def test_no_nans(self):
        fp = compute_morgan_fingerprint("c1ccccc1")
        assert not np.any(np.isnan(fp))

    def test_binary_values(self):
        fp = compute_morgan_fingerprint("c1ccccc1")
        assert set(np.unique(fp)).issubset({0, 1})

    def test_invalid_smiles(self):
        fp = compute_morgan_fingerprint("INVALID_SMILES")
        assert fp is None

    def test_different_molecules_different_fps(self):
        fp1 = compute_morgan_fingerprint("c1ccccc1")  # benzene
        fp2 = compute_morgan_fingerprint("C1CCCCC1")  # cyclohexane
        assert not np.array_equal(fp1, fp2)


class TestDescriptors:
    """Tests for molecular descriptor computation."""

    def test_valid_molecule(self):
        desc = compute_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")  # aspirin
        assert desc is not None
        assert "mw" in desc
        assert "logp" in desc
        assert "tpsa" in desc
        assert "hbd" in desc
        assert "hba" in desc
        assert "rotatable_bonds" in desc

    def test_reasonable_values(self):
        desc = compute_descriptors("O")  # water
        assert desc["mw"] > 0
        assert desc["num_atoms"] == 1  # one heavy atom

    def test_invalid_smiles(self):
        desc = compute_descriptors("INVALID")
        assert desc is None
