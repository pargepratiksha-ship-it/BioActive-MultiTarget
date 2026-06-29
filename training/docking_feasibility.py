"""
Docking Feasibility Assessment Module.

Determines whether molecular docking is scientifically appropriate for a given
ligand BEFORE attempting docking. This prevents wasted computation and, more
importantly, prevents scientifically meaningless docking scores from entering
the prediction pipeline.

Scientific basis:
- AutoDock Vina was validated on drug-like molecules with ≤32 rotatable bonds
  (Trott & Olson, J. Comput. Chem., 2010).
- Classical docking scoring functions lose predictive accuracy for molecules
  >2000 Da (Warren et al., J. Med. Chem., 2006).
- Peptides beyond ~8-10 residues have conformational entropy that overwhelms
  rigid-receptor docking (London et al., Proteins, 2013).
- RDKit 3D embedding failure indicates the molecule is too complex for reliable
  conformer generation, which is a prerequisite for docking.

All thresholds are configurable and documented with their scientific rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — all thresholds with scientific rationale
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FeasibilityThresholds:
    """Configurable thresholds for docking feasibility assessment.

    Each threshold has a scientific rationale documented in the field metadata.
    Adjust these based on your docking engine and validation studies.
    """

    max_rotatable_bonds: int = 32
    """Maximum rotatable bonds for reliable Vina sampling.
    Rationale: Vina's stochastic search becomes unreliable above this limit.
    Each rotatable bond adds 3 degrees of freedom to the search space.
    At 32 bonds, the conformational space is ~3^32 ≈ 10^15 states.
    Reference: Forli et al., Nat. Protoc., 2016; Vina validation set ≤32."""

    max_molecular_weight: float = 2000.0
    """Maximum molecular weight (Da) for reliable scoring function accuracy.
    Rationale: Empirical scoring functions (Vina's hybrid) were parameterized
    on drug-like molecules (150-800 Da). Above ~2000 Da, the relationship
    between score and binding affinity degrades significantly.
    Reference: Warren et al., J. Med. Chem., 2006."""

    max_heavy_atoms: int = 150
    """Maximum heavy atom count for tractable docking search space.
    Rationale: Related to molecular complexity. The number of interacting
    atom pairs scales as N², making scoring unreliable for very large ligands.
    150 heavy atoms corresponds roughly to a 15-residue peptide."""

    max_peptide_length: int = 10
    """Maximum peptide residue count for meaningful rigid-receptor docking.
    Rationale: Peptides >10 residues adopt many backbone conformations in
    solution. Rigid-receptor docking cannot sample this ensemble adequately.
    Flexible peptide docking (e.g., Rosetta FlexPepDock) is needed beyond this.
    Reference: London et al., Proteins, 2013."""

    min_molecular_weight: float = 50.0
    """Minimum molecular weight — below this, the molecule is too small to have
    meaningful binding interactions with a protein active site."""

    require_3d_embedding: bool = True
    """Whether to require successful 3D coordinate generation as a prerequisite.
    Rationale: If RDKit cannot generate a plausible 3D conformer, the docking
    input geometry is unreliable and results will be meaningless."""

    rotatable_bonds_caution: int = 25
    """Threshold above which docking is still attempted but with reduced
    exhaustiveness and a caution flag. Results in this range should be
    interpreted with lower confidence."""


# Default thresholds instance
DEFAULT_THRESHOLDS = FeasibilityThresholds()


# ═══════════════════════════════════════════════════════════════════════════════
# Result types
# ═══════════════════════════════════════════════════════════════════════════════

class FeasibilityRecommendation(str, Enum):
    """Docking recommendation based on feasibility assessment."""
    PROCEED = "proceed"
    """Molecule is well within validated docking parameters."""

    CAUTION = "caution"
    """Molecule is borderline — docking will proceed with reduced parameters
    and results should be interpreted with lower confidence."""

    SKIP = "skip"
    """Molecule exceeds validated docking parameters. Docking would produce
    scientifically unreliable results and should not be attempted."""


@dataclass
class FeasibilityResult:
    """Complete result of a docking feasibility assessment.

    Attributes:
        is_suitable: Whether docking should be attempted (proceed or caution).
        recommendation: Granular recommendation (proceed/caution/skip).
        reasons: Human-readable explanations for the decision.
        metrics: All computed molecular metrics used in the assessment.
        skip_reasons: Specific reasons why docking was deemed inappropriate
            (empty if suitable).
        caution_reasons: Reasons for reduced confidence (empty if fully suitable).
        confidence: Assessment confidence (1.0 = clear decision, lower = borderline).
    """
    is_suitable: bool
    recommendation: FeasibilityRecommendation
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    skip_reasons: list[str] = field(default_factory=list)
    caution_reasons: list[str] = field(default_factory=list)
    confidence: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Core Assessment Functions
# ═══════════════════════════════════════════════════════════════════════════════

def assess_docking_feasibility(
    smiles: str,
    is_peptide: bool = False,
    peptide_length: int | None = None,
    thresholds: FeasibilityThresholds | None = None,
) -> FeasibilityResult:
    """Assess whether molecular docking is scientifically appropriate for a ligand.

    This function inspects the molecule's structural properties and determines
    whether classical rigid-receptor docking (AutoDock Vina) can produce
    scientifically meaningful results.

    Args:
        smiles: SMILES string of the ligand (must be valid).
        is_peptide: Whether the ligand is a peptide (enables peptide-specific checks).
        peptide_length: Number of amino acid residues (if peptide). If None and
            is_peptide is True, estimated from molecular weight.
        thresholds: Custom thresholds. Uses DEFAULT_THRESHOLDS if None.

    Returns:
        FeasibilityResult with recommendation, reasons, and molecular metrics.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return FeasibilityResult(
            is_suitable=False,
            recommendation=FeasibilityRecommendation.SKIP,
            reasons=["Invalid SMILES — cannot parse molecule"],
            skip_reasons=["invalid_smiles"],
            confidence=1.0,
        )

    # ── Compute molecular metrics ──
    mol_h = Chem.AddHs(mol)
    n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol_h)
    mw = Descriptors.ExactMolWt(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    n_atoms_total = mol_h.GetNumAtoms()

    # Estimate peptide length from MW if not provided
    if is_peptide and peptide_length is None:
        # Average amino acid residue MW ≈ 110 Da (minus water)
        peptide_length = max(1, round((mw + 18.0) / 110.0))

    metrics = {
        "rotatable_bonds": n_rotatable,
        "molecular_weight": round(mw, 2),
        "heavy_atoms": n_heavy,
        "total_atoms_with_h": n_atoms_total,
        "is_peptide": is_peptide,
        "peptide_length": peptide_length,
    }

    # ── Evaluate criteria ──
    skip_reasons: list[str] = []
    caution_reasons: list[str] = []
    reasons: list[str] = []

    # Criterion 1: Rotatable bonds (primary limitation for Vina)
    if n_rotatable > thresholds.max_rotatable_bonds:
        skip_reasons.append(
            f"Rotatable bonds ({n_rotatable}) exceed maximum ({thresholds.max_rotatable_bonds}). "
            f"Vina's stochastic search cannot reliably sample this conformational space."
        )
    elif n_rotatable > thresholds.rotatable_bonds_caution:
        caution_reasons.append(
            f"Rotatable bonds ({n_rotatable}) in caution zone "
            f"({thresholds.rotatable_bonds_caution}-{thresholds.max_rotatable_bonds}). "
            f"Reduced exhaustiveness recommended."
        )
    else:
        reasons.append(
            f"Rotatable bonds ({n_rotatable}) within reliable range (≤{thresholds.rotatable_bonds_caution})."
        )

    # Criterion 2: Molecular weight
    if mw > thresholds.max_molecular_weight:
        skip_reasons.append(
            f"Molecular weight ({mw:.0f} Da) exceeds maximum ({thresholds.max_molecular_weight:.0f} Da). "
            f"Scoring function accuracy degrades for molecules this large."
        )
    elif mw < thresholds.min_molecular_weight:
        skip_reasons.append(
            f"Molecular weight ({mw:.0f} Da) below minimum ({thresholds.min_molecular_weight:.0f} Da). "
            f"Molecule too small for meaningful protein-ligand interactions."
        )
    else:
        reasons.append(f"Molecular weight ({mw:.0f} Da) within valid range.")

    # Criterion 3: Heavy atom count
    if n_heavy > thresholds.max_heavy_atoms:
        skip_reasons.append(
            f"Heavy atoms ({n_heavy}) exceed maximum ({thresholds.max_heavy_atoms}). "
            f"Search space complexity too high for reliable docking."
        )

    # Criterion 4: Peptide length (only if peptide)
    if is_peptide and peptide_length is not None:
        if peptide_length > thresholds.max_peptide_length:
            skip_reasons.append(
                f"Peptide length ({peptide_length} residues) exceeds maximum "
                f"({thresholds.max_peptide_length} residues). "
                f"Rigid-receptor docking cannot adequately sample backbone flexibility "
                f"for peptides this long. Consider FlexPepDock or MD-based approaches."
            )
        else:
            reasons.append(
                f"Peptide length ({peptide_length} residues) within dockable range."
            )

    # Criterion 5: 3D embedding test (optional but recommended)
    embedding_ok = True
    if thresholds.require_3d_embedding:
        embedding_ok = _test_3d_embedding(mol_h)
        metrics["3d_embedding_success"] = embedding_ok
        if not embedding_ok:
            skip_reasons.append(
                "3D coordinate generation failed. RDKit cannot produce a plausible "
                "starting conformer, making docking input geometry unreliable."
            )
        else:
            reasons.append("3D embedding successful.")

    # ── Determine recommendation ──
    if skip_reasons:
        recommendation = FeasibilityRecommendation.SKIP
        is_suitable = False
        # Confidence is high when multiple criteria agree on skip
        confidence = min(1.0, 0.7 + 0.1 * len(skip_reasons))
    elif caution_reasons:
        recommendation = FeasibilityRecommendation.CAUTION
        is_suitable = True
        confidence = 0.7
    else:
        recommendation = FeasibilityRecommendation.PROCEED
        is_suitable = True
        confidence = 1.0

    all_reasons = reasons + caution_reasons + skip_reasons

    return FeasibilityResult(
        is_suitable=is_suitable,
        recommendation=recommendation,
        reasons=all_reasons,
        metrics=metrics,
        skip_reasons=skip_reasons,
        caution_reasons=caution_reasons,
        confidence=confidence,
    )


def get_adaptive_docking_parameters(
    feasibility: FeasibilityResult,
    base_exhaustiveness: int = 8,
) -> dict:
    """Determine docking parameters based on feasibility assessment.

    For molecules in the CAUTION zone, reduces exhaustiveness and modes
    to avoid timeouts while still attempting docking.

    Args:
        feasibility: Result from assess_docking_feasibility().
        base_exhaustiveness: Default exhaustiveness for normal molecules.

    Returns:
        Dict with keys: exhaustiveness, num_modes, timeout_sec.
    """
    n_rot = feasibility.metrics.get("rotatable_bonds", 0)

    if feasibility.recommendation == FeasibilityRecommendation.SKIP:
        # Should not be called for SKIP, but return minimal params as safety
        return {"exhaustiveness": 1, "num_modes": 1, "timeout_sec": 60}

    if feasibility.recommendation == FeasibilityRecommendation.CAUTION:
        # Borderline molecules (25-32 rot bonds) — minimal sampling
        if n_rot > 28:
            return {"exhaustiveness": 1, "num_modes": 1, "timeout_sec": 600}
        else:
            return {"exhaustiveness": 1, "num_modes": 2, "timeout_sec": 480}

    # PROCEED — parameters scaled by complexity.
    # Vina search space grows exponentially with rotatable bonds, so
    # exhaustiveness MUST be 1 for high-flexibility molecules.
    if n_rot > 18:
        return {"exhaustiveness": 1, "num_modes": 3, "timeout_sec": 600}
    elif n_rot > 12:
        return {"exhaustiveness": 2, "num_modes": 3, "timeout_sec": 300}
    elif n_rot > 6:
        return {"exhaustiveness": 4, "num_modes": 5, "timeout_sec": 180}
    else:
        return {"exhaustiveness": base_exhaustiveness, "num_modes": 5, "timeout_sec": 120}


# ═══════════════════════════════════════════════════════════════════════════════
# Internal Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _test_3d_embedding(mol_h: Chem.Mol) -> bool:
    """Test whether RDKit can generate a 3D conformer for this molecule.

    Uses ETKDGv3 with fallback to random coordinates. Returns True if
    at least one method succeeds.
    """
    # Work on a copy to avoid mutating the input
    mol_copy = Chem.RWMol(mol_h)
    try:
        params = AllChem.ETKDGv3()
        result = AllChem.EmbedMolecule(mol_copy, params)
        if result != -1:
            return True
        # Fallback: random coordinates
        params2 = AllChem.ETKDGv3()
        params2.useRandomCoords = True
        result = AllChem.EmbedMolecule(mol_copy, params2)
        return result != -1
    except Exception:
        return False
