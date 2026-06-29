"""Tests for the generic pipeline module functions."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from training.pipeline import (
    MORGAN_NBITS,
    canonicalize_smiles,
    compute_descriptors,
    compute_morgan_fingerprint,
    get_murcko_scaffold,
    scaffold_split,
    smiles_to_fingerprint,
)


class TestCanonicalizeSmiles:
    """Tests for SMILES canonicalization."""

    def test_valid_smiles(self):
        result = canonicalize_smiles("C(=O)O")
        assert result is not None
        assert isinstance(result, str)

    def test_canonical_form(self):
        s1 = canonicalize_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")
        s2 = canonicalize_smiles("O=C(O)c1ccccc1OC(C)=O")
        assert s1 == s2

    def test_invalid_smiles(self):
        assert canonicalize_smiles("NOT_A_MOLECULE") is None

    def test_empty_string(self):
        assert canonicalize_smiles("") is None

    def test_none_input(self):
        assert canonicalize_smiles(None) is None

    def test_numeric_input(self):
        assert canonicalize_smiles(123) is None


class TestMurckoScaffold:
    """Tests for Murcko scaffold computation."""

    def test_simple_molecule(self):
        scaffold = get_murcko_scaffold("c1ccccc1")
        assert scaffold is not None
        assert isinstance(scaffold, str)

    def test_returns_string_for_invalid(self):
        result = get_murcko_scaffold("INVALID")
        assert result == "INVALID"


class TestScaffoldSplit:
    """Tests for scaffold-based train/test splitting."""

    def test_split_sizes(self):
        df = pd.DataFrame({
            "canonical_smiles": [
                "c1ccccc1", "c1ccc(O)cc1", "c1ccc(N)cc1",
                "C1CCCCC1", "C1CCC(O)CC1", "CC(=O)O",
                "CCCC", "CCCCC", "CCCCCC", "CCCCCCC",
            ],
            "active": [1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        })
        train, test = scaffold_split(df, test_fraction=0.2, seed=42)
        assert len(train) + len(test) == len(df)
        assert len(test) > 0
        assert len(train) > 0

    def test_no_overlap(self):
        df = pd.DataFrame({
            "canonical_smiles": [f"CCCC{'C' * i}" for i in range(20)],
            "active": [i % 2 for i in range(20)],
        })
        train, test = scaffold_split(df, test_fraction=0.2, seed=42)
        train_smiles = set(train["canonical_smiles"])
        test_smiles = set(test["canonical_smiles"])
        assert len(train_smiles & test_smiles) == 0


class TestMorganFingerprint:
    """Tests for Morgan fingerprint computation."""

    def test_valid_smiles(self):
        fp = compute_morgan_fingerprint("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert fp is not None
        assert fp.shape == (MORGAN_NBITS,)

    def test_no_nans(self):
        fp = compute_morgan_fingerprint("c1ccccc1")
        assert not np.any(np.isnan(fp))

    def test_binary_values(self):
        fp = compute_morgan_fingerprint("c1ccccc1")
        assert set(np.unique(fp)).issubset({0, 1})

    def test_invalid_smiles(self):
        fp = compute_morgan_fingerprint("INVALID_SMILES")
        assert fp is None

    def test_different_molecules(self):
        fp1 = compute_morgan_fingerprint("c1ccccc1")
        fp2 = compute_morgan_fingerprint("C1CCCCC1")
        assert not np.array_equal(fp1, fp2)


class TestDescriptors:
    """Tests for molecular descriptor computation."""

    def test_valid_molecule(self):
        desc = compute_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")
        assert desc is not None
        assert "mw" in desc
        assert "logp" in desc
        assert "tpsa" in desc
        assert "hbd" in desc
        assert "hba" in desc
        assert "rotatable_bonds" in desc

    def test_reasonable_values(self):
        desc = compute_descriptors("O")
        assert desc["mw"] > 0
        assert desc["num_atoms"] == 1

    def test_invalid_smiles(self):
        desc = compute_descriptors("INVALID")
        assert desc is None


class TestSmilesToFingerprint:
    """Tests for the benchmark fingerprint utility."""

    def test_valid_compound(self):
        fp = smiles_to_fingerprint("c1ccccc1")
        assert fp is not None
        assert fp.shape == (MORGAN_NBITS,)

    def test_invalid_compound(self):
        fp = smiles_to_fingerprint("NOT_VALID")
        assert fp is None
