"""Tests for all target benchmark compound panels.

Validates that every benchmark compound across all targets has:
    - Valid, parseable SMILES (RDKit can interpret)
    - Correct category assignment (potent, weak, or unrelated)
    - Minimum panel sizes for statistical robustness
    - No duplicate compound names within a target
    - Valid fingerprint generation
"""

import numpy as np
import pytest
from rdkit import Chem

from training.pipeline import smiles_to_fingerprint
from training.target_configs import ALL_TARGETS


class TestAllBenchmarkPanels:
    """Cross-target benchmark validation tests."""

    @pytest.fixture(params=list(ALL_TARGETS.keys()))
    def target_config(self, request):
        """Parametrize tests across all targets."""
        return ALL_TARGETS[request.param]

    def test_all_smiles_valid(self, target_config):
        """Every benchmark compound must have a parseable SMILES."""
        for compound in target_config["benchmark_compounds"]:
            mol = Chem.MolFromSmiles(compound["smiles"])
            assert mol is not None, (
                f"Invalid SMILES for {compound['name']} in "
                f"{target_config['name']}: {compound['smiles']}"
            )

    def test_categories_present(self, target_config):
        """Panel must include potent, weak, and unrelated categories."""
        categories = {c["category"] for c in target_config["benchmark_compounds"]}
        assert "potent" in categories, f"Missing 'potent' in {target_config['name']}"
        assert "weak" in categories, f"Missing 'weak' in {target_config['name']}"
        assert "unrelated" in categories, f"Missing 'unrelated' in {target_config['name']}"

    def test_minimum_potent_inhibitors(self, target_config):
        """Must have at least 3 known potent inhibitors."""
        potent = [c for c in target_config["benchmark_compounds"] if c["category"] == "potent"]
        assert len(potent) >= 3, (
            f"Only {len(potent)} potent inhibitors in {target_config['name']} "
            f"(minimum 3 required)"
        )

    def test_minimum_unrelated_compounds(self, target_config):
        """Must have at least 3 unrelated compounds."""
        unrelated = [c for c in target_config["benchmark_compounds"] if c["category"] == "unrelated"]
        assert len(unrelated) >= 3, f"Only {len(unrelated)} unrelated in {target_config['name']}"

    def test_no_duplicate_names(self, target_config):
        """No duplicate compound names within a target."""
        names = [c["name"] for c in target_config["benchmark_compounds"]]
        assert len(names) == len(set(names)), (
            f"Duplicate compound names in {target_config['name']}"
        )

    def test_fingerprint_generation(self, target_config):
        """All benchmark compounds must produce valid fingerprints."""
        for compound in target_config["benchmark_compounds"]:
            fp = smiles_to_fingerprint(compound["smiles"])
            assert fp is not None, (
                f"Fingerprint failed for {compound['name']} in {target_config['name']}"
            )
            assert fp.shape == (2048,)


class TestTargetConfigs:
    """Validate target configuration structure."""

    @pytest.fixture(params=list(ALL_TARGETS.keys()))
    def config(self, request):
        return ALL_TARGETS[request.param]

    def test_required_fields(self, config):
        """Every config must have required fields."""
        required = ["chembl_id", "name", "display_name", "activity_types",
                    "activity_threshold_nM", "description", "benchmark_compounds"]
        for field in required:
            assert field in config, f"Missing '{field}' in {config.get('name', 'unknown')}"

    def test_chembl_id_format(self, config):
        """ChEMBL ID must start with 'CHEMBL'."""
        chembl_ids = config["chembl_id"]
        if isinstance(chembl_ids, str):
            chembl_ids = [chembl_ids]
        for cid in chembl_ids:
            assert cid.startswith("CHEMBL"), (
                f"Invalid ChEMBL ID: {cid}"
            )

    def test_threshold_positive(self, config):
        """Activity threshold must be positive."""
        assert config["activity_threshold_nM"] > 0

    def test_activity_types_nonempty(self, config):
        """Must have at least one activity type."""
        assert len(config["activity_types"]) > 0
