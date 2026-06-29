"""
Peptide Physicochemical Scoring Module.

Computes compatibility scores for peptides against enzyme targets based on
known physicochemical preferences of each target's active site.

This is NOT ML-based — it uses domain knowledge from published literature
about what properties make a peptide likely to inhibit each target.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

# Amino acid properties (hydrophobicity index, Kyte-Doolittle scale)
AA_HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Amino acid molecular weights
AA_MW = {
    "A": 89.1, "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2,
    "E": 147.1, "Q": 146.2, "G": 75.0, "H": 155.2, "I": 131.2,
    "L": 131.2, "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1,
    "S": 105.1, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1,
}

# Target-specific profiles derived from published literature
# Each profile defines optimal peptide properties for inhibition
TARGET_PROFILES = {
    "ace": {
        "description": "ACE-inhibitory peptides: C-terminal hydrophobic/aromatic AAs, "
                       "proline-rich, 2-12 residues, moderate hydrophobicity",
        "optimal_length": (2, 12),
        "preferred_c_terminal": set("PFYWILVM"),  # Hydrophobic/aromatic C-terminus (incl. Met)
        "preferred_residues": set("PFYWILVKM"),  # Pro, Phe, Trp, Tyr, Ile, Leu, Val, Lys, Met
        "hydrophobicity_range": (-1.0, 2.5),  # Mean hydrophobicity per residue
        "mw_range": (200, 1500),
        "weights": {
            "c_terminal": 0.30,
            "residue_composition": 0.25,
            "hydrophobicity": 0.20,
            "length": 0.15,
            "mw": 0.10,
        },
    },
    "dpp4": {
        "description": "DPP-IV inhibitory peptides: N-terminal X-Pro or X-Ala dipeptides, "
                       "2-8 residues, proline at position 2",
        "optimal_length": (2, 8),
        "preferred_n_terminal_p2": set("PA"),  # Pro or Ala at position 2
        "preferred_residues": set("PAGILWF"),
        "hydrophobicity_range": (-0.5, 3.0),
        "mw_range": (150, 1000),
        "weights": {
            "n_terminal_p2": 0.35,
            "residue_composition": 0.25,
            "hydrophobicity": 0.20,
            "length": 0.10,
            "mw": 0.10,
        },
    },
    "amylase": {
        "description": "Alpha-amylase inhibitory peptides: aromatic residues, moderate length, "
                       "ability to interact with active site subsites",
        "optimal_length": (3, 10),
        "preferred_residues": set("FYWRHKP"),  # Aromatic + charged
        "hydrophobicity_range": (-2.0, 1.5),
        "mw_range": (300, 1200),
        "weights": {
            "residue_composition": 0.35,
            "hydrophobicity": 0.25,
            "length": 0.20,
            "mw": 0.20,
        },
    },
    "glucosidase": {
        "description": "Alpha-glucosidase inhibitory peptides: similar to amylase, "
                       "aromatic and basic residues, sugar-mimicking properties",
        "optimal_length": (3, 10),
        "preferred_residues": set("FYWRHKP"),
        "hydrophobicity_range": (-2.0, 1.5),
        "mw_range": (300, 1200),
        "weights": {
            "residue_composition": 0.35,
            "hydrophobicity": 0.25,
            "length": 0.20,
            "mw": 0.20,
        },
    },
    "lipase": {
        "description": "Lipase inhibitory peptides: hydrophobic residues, amphipathic, "
                       "ability to interact with lid domain, 4-12 residues",
        "optimal_length": (4, 12),
        "preferred_residues": set("ILVFMWPA"),  # Hydrophobic
        "hydrophobicity_range": (0.5, 4.0),
        "mw_range": (400, 1500),
        "weights": {
            "residue_composition": 0.35,
            "hydrophobicity": 0.30,
            "length": 0.20,
            "mw": 0.15,
        },
    },
    "egfr": {
        "description": "EGFR-targeting peptides: not well-established for food peptides",
        "optimal_length": (5, 15),
        "preferred_residues": set("FYWRH"),
        "hydrophobicity_range": (-1.0, 2.0),
        "mw_range": (500, 2000),
        "weights": {
            "residue_composition": 0.30,
            "hydrophobicity": 0.30,
            "length": 0.20,
            "mw": 0.20,
        },
    },
    "her2": {
        "description": "HER2-targeting peptides: not well-established for food peptides",
        "optimal_length": (5, 15),
        "preferred_residues": set("FYWRH"),
        "hydrophobicity_range": (-1.0, 2.0),
        "mw_range": (500, 2000),
        "weights": {
            "residue_composition": 0.30,
            "hydrophobicity": 0.30,
            "length": 0.20,
            "mw": 0.20,
        },
    },
    "vegfr2": {
        "description": "VEGFR2-targeting peptides: not well-established for food peptides",
        "optimal_length": (5, 15),
        "preferred_residues": set("FYWRH"),
        "hydrophobicity_range": (-1.0, 2.0),
        "mw_range": (500, 2000),
        "weights": {
            "residue_composition": 0.30,
            "hydrophobicity": 0.30,
            "length": 0.20,
            "mw": 0.20,
        },
    },
    "braf": {
        "description": "BRAF-targeting peptides: not established for food peptides",
        "optimal_length": (5, 15),
        "preferred_residues": set("FYWRH"),
        "hydrophobicity_range": (-1.0, 2.0),
        "mw_range": (500, 2000),
        "weights": {
            "residue_composition": 0.30,
            "hydrophobicity": 0.30,
            "length": 0.20,
            "mw": 0.20,
        },
    },
    "cdk2": {
        "description": "CDK2-targeting peptides: not established for food peptides",
        "optimal_length": (5, 15),
        "preferred_residues": set("FYWRH"),
        "hydrophobicity_range": (-1.0, 2.0),
        "mw_range": (500, 2000),
        "weights": {
            "residue_composition": 0.30,
            "hydrophobicity": 0.30,
            "length": 0.20,
            "mw": 0.20,
        },
    },
}


def compute_peptide_descriptors(sequence: str) -> dict | None:
    """Compute physicochemical descriptors for a peptide sequence.

    Args:
        sequence: Peptide sequence in 1-letter codes.

    Returns:
        Dictionary of descriptors or None if invalid.
    """
    sequence = sequence.strip().upper()
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    if not sequence or not all(c in valid_aa for c in sequence):
        return None

    length = len(sequence)
    mw = sum(AA_MW.get(aa, 0) for aa in sequence) - (length - 1) * 18.015  # Peptide bonds

    # Mean hydrophobicity
    hydrophobicities = [AA_HYDROPHOBICITY.get(aa, 0) for aa in sequence]
    mean_hydrophobicity = np.mean(hydrophobicities)

    # Amino acid composition
    aa_counts = {aa: sequence.count(aa) / length for aa in valid_aa}

    # Charge at pH 7
    pos_charged = sequence.count("R") + sequence.count("K") + sequence.count("H") * 0.1
    neg_charged = sequence.count("D") + sequence.count("E")
    net_charge = pos_charged - neg_charged

    # Aromatic content
    aromatic_fraction = sum(1 for aa in sequence if aa in "FWY") / length

    # Isoelectric point estimation (simplified)
    pi_estimate = 6.0 + (pos_charged - neg_charged) * 0.5

    return {
        "sequence": sequence,
        "length": length,
        "mw": mw,
        "mean_hydrophobicity": mean_hydrophobicity,
        "net_charge": net_charge,
        "aromatic_fraction": aromatic_fraction,
        "pi_estimate": pi_estimate,
        "c_terminal": sequence[-1] if sequence else "",
        "n_terminal": sequence[0] if sequence else "",
        "position_2": sequence[1] if len(sequence) > 1 else "",
        "aa_composition": aa_counts,
    }


def score_length(length: int, optimal_range: tuple[int, int]) -> float:
    """Score peptide length vs optimal range (0-1)."""
    low, high = optimal_range
    if low <= length <= high:
        return 1.0
    elif length < low:
        return max(0, 1.0 - (low - length) * 0.2)
    else:
        return max(0, 1.0 - (length - high) * 0.15)


def score_mw(mw: float, mw_range: tuple[float, float]) -> float:
    """Score molecular weight vs optimal range (0-1)."""
    low, high = mw_range
    if low <= mw <= high:
        return 1.0
    elif mw < low:
        return max(0, 1.0 - (low - mw) / 200)
    else:
        return max(0, 1.0 - (mw - high) / 500)


def score_hydrophobicity(mean_h: float, h_range: tuple[float, float]) -> float:
    """Score hydrophobicity vs optimal range (0-1)."""
    low, high = h_range
    if low <= mean_h <= high:
        return 1.0
    elif mean_h < low:
        return max(0, 1.0 - (low - mean_h) * 0.3)
    else:
        return max(0, 1.0 - (mean_h - high) * 0.3)


def score_residue_composition(sequence: str, preferred: set) -> float:
    """Score fraction of preferred residues (0-1)."""
    if not sequence:
        return 0
    preferred_count = sum(1 for aa in sequence if aa in preferred)
    return preferred_count / len(sequence)


def compute_compatibility_score(sequence: str, target: str) -> dict | None:
    """Compute physicochemical compatibility score for a peptide-target pair.

    Args:
        sequence: Peptide sequence (1-letter codes).
        target: Target name (lowercase, e.g., 'ace', 'lipase').

    Returns:
        Dictionary with component scores and overall compatibility, or None.
    """
    target_key = target.lower().replace("-", "").replace(" ", "")
    # Map display names
    name_map = {
        "dpp4": "dpp4", "alphaamylase": "amylase", "alphaglucosidase": "glucosidase",
        "pancreaticlipase": "lipase", "ace": "ace", "egfr": "egfr",
        "her2": "her2", "vegfr2": "vegfr2", "braf": "braf", "cdk2": "cdk2",
        "amylase": "amylase", "glucosidase": "glucosidase", "lipase": "lipase",
    }
    profile_key = name_map.get(target_key)
    if profile_key is None:
        return None

    profile = TARGET_PROFILES.get(profile_key)
    if profile is None:
        return None

    descriptors = compute_peptide_descriptors(sequence)
    if descriptors is None:
        return None

    scores = {}
    weights = profile["weights"]

    # Length score
    scores["length"] = score_length(descriptors["length"], profile["optimal_length"])

    # MW score
    scores["mw"] = score_mw(descriptors["mw"], profile["mw_range"])

    # Hydrophobicity score
    scores["hydrophobicity"] = score_hydrophobicity(
        descriptors["mean_hydrophobicity"], profile["hydrophobicity_range"]
    )

    # Residue composition score
    scores["residue_composition"] = score_residue_composition(
        descriptors["sequence"], profile["preferred_residues"]
    )

    # Target-specific scores
    if profile_key == "ace" and "c_terminal" in weights:
        scores["c_terminal"] = 1.0 if descriptors["c_terminal"] in profile["preferred_c_terminal"] else 0.2

    if profile_key == "dpp4" and "n_terminal_p2" in weights:
        scores["n_terminal_p2"] = 1.0 if descriptors["position_2"] in profile["preferred_n_terminal_p2"] else 0.2

    # Compute weighted overall score
    overall = 0.0
    total_weight = 0.0
    for component, weight in weights.items():
        if component in scores:
            overall += scores[component] * weight
            total_weight += weight

    if total_weight > 0:
        overall /= total_weight

    return {
        "sequence": descriptors["sequence"],
        "target": profile_key,
        "compatibility_score": float(overall),
        "component_scores": scores,
        "descriptors": descriptors,
        "profile_description": profile["description"],
    }
