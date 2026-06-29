"""Tests for DPP4 data processing pipeline."""

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from training.process_dpp4 import canonicalize_smiles, get_murcko_scaffold, scaffold_split


class TestCanonicalizeSmiles:
    """Tests for SMILES canonicalization."""

    def test_valid_smiles(self):
        result = canonicalize_smiles("C(=O)O")
        assert result is not None
        assert isinstance(result, str)

    def test_canonical_form(self):
        # Different representations of aspirin should give the same canonical SMILES
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
        scaffold = get_murcko_scaffold("c1ccccc1")  # benzene
        assert scaffold is not None
        assert isinstance(scaffold, str)

    def test_returns_string_for_invalid(self):
        # Should return the original SMILES if scaffolding fails
        result = get_murcko_scaffold("INVALID")
        assert result == "INVALID"


class TestScaffoldSplit:
    """Tests for scaffold-based train/test splitting."""

    def test_split_sizes(self):
        df = pd.DataFrame({
            "canonical_smiles": [
                "c1ccccc1",
                "c1ccc(O)cc1",
                "c1ccc(N)cc1",
                "C1CCCCC1",
                "C1CCC(O)CC1",
                "CC(=O)O",
                "CCCC",
                "CCCCC",
                "CCCCCC",
                "CCCCCCC",
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
