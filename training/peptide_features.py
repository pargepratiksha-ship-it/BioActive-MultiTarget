"""
Peptide Feature Engineering for Bioactivity Prediction.

Converts peptide sequences into numerical feature vectors suitable for ML models.
Uses physicochemical descriptors, composition features, and structural properties
instead of Morgan fingerprints (which are inappropriate for peptides).

Features are based on:
- Amino acid composition (AAC)
- Dipeptide composition (DPC)
- Physicochemical properties (hydrophobicity, charge, MW, pI)
- Structural/positional features (N-terminal, C-terminal residues)
- Target-specific motif features
"""

import numpy as np
import pandas as pd
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# Amino Acid Properties
# ═══════════════════════════════════════════════════════════════════════════════

STANDARD_AAS = "ACDEFGHIKLMNPQRSTVWY"

# Kyte-Doolittle hydrophobicity
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Molecular weight of residues (Da)
AA_MW = {
    "A": 89.1, "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2,
    "E": 147.1, "Q": 146.2, "G": 75.0, "H": 155.2, "I": 131.2,
    "L": 131.2, "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1,
    "S": 105.1, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1,
}

# Charge at pH 7
AA_CHARGE = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "E": -1, "Q": 0, "G": 0, "H": 0.1, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}

# Bulkiness (Zimmerman scale)
AA_BULK = {
    "A": 11.50, "R": 14.28, "N": 12.82, "D": 11.68, "C": 13.46,
    "E": 13.57, "Q": 14.45, "G": 3.40, "H": 13.69, "I": 21.40,
    "L": 21.40, "K": 15.71, "M": 16.25, "F": 19.80, "P": 17.43,
    "S": 9.47, "T": 15.77, "W": 21.67, "Y": 18.03, "V": 21.57,
}

# Flexibility (Bhaskaran-Ponnuswamy scale)
AA_FLEXIBILITY = {
    "A": 0.357, "R": 0.529, "N": 0.463, "D": 0.511, "C": 0.346,
    "E": 0.497, "Q": 0.493, "G": 0.544, "H": 0.323, "I": 0.462,
    "L": 0.365, "K": 0.466, "M": 0.295, "F": 0.314, "P": 0.509,
    "S": 0.507, "T": 0.444, "W": 0.305, "Y": 0.420, "V": 0.386,
}

# Whether an amino acid is aromatic
AA_AROMATIC = {"F", "W", "Y", "H"}
# Whether branched aliphatic
AA_ALIPHATIC = {"I", "L", "V", "A"}


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Computation Functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_aac(sequence: str) -> np.ndarray:
    """Amino acid composition — fraction of each AA in sequence (20 features)."""
    length = len(sequence)
    if length == 0:
        return np.zeros(20)
    counts = np.zeros(20)
    for aa in sequence:
        idx = STANDARD_AAS.find(aa)
        if idx >= 0:
            counts[idx] += 1
    return counts / length


def compute_dpc(sequence: str) -> np.ndarray:
    """Dipeptide composition — fraction of each dipeptide (400 features)."""
    length = len(sequence) - 1
    if length <= 0:
        return np.zeros(400)
    counts = np.zeros(400)
    for i in range(length):
        idx1 = STANDARD_AAS.find(sequence[i])
        idx2 = STANDARD_AAS.find(sequence[i + 1])
        if idx1 >= 0 and idx2 >= 0:
            counts[idx1 * 20 + idx2] += 1
    return counts / length


def compute_physicochemical(sequence: str) -> np.ndarray:
    """Global physicochemical properties (12 features)."""
    length = len(sequence)
    if length == 0:
        return np.zeros(12)

    hydro_values = [HYDROPHOBICITY.get(aa, 0) for aa in sequence]
    charge_values = [AA_CHARGE.get(aa, 0) for aa in sequence]
    bulk_values = [AA_BULK.get(aa, 0) for aa in sequence]
    flex_values = [AA_FLEXIBILITY.get(aa, 0) for aa in sequence]

    mw = sum(AA_MW.get(aa, 0) for aa in sequence) - (length - 1) * 18.0  # peptide bonds

    features = [
        length,                                     # 1. Length
        mw,                                         # 2. Molecular weight
        np.mean(hydro_values),                      # 3. Mean hydrophobicity
        np.std(hydro_values) if length > 1 else 0,  # 4. Hydrophobicity variation
        np.sum(charge_values),                      # 5. Net charge
        sum(1 for aa in sequence if aa in AA_AROMATIC) / length,  # 6. Aromatic fraction
        sum(1 for aa in sequence if aa in AA_ALIPHATIC) / length,  # 7. Aliphatic fraction
        np.mean(bulk_values),                       # 8. Mean bulkiness
        np.mean(flex_values),                       # 9. Mean flexibility
        sequence.count("P") / length,               # 10. Proline fraction
        sequence.count("G") / length,               # 11. Glycine fraction
        max(hydro_values) - min(hydro_values) if length > 1 else 0,  # 12. Hydro range
    ]
    return np.array(features, dtype=np.float64)


def compute_terminal_features(sequence: str) -> np.ndarray:
    """N-terminal and C-terminal residue features (40 features).

    One-hot encoding of first and last residue (20 + 20).
    Critical for DPP4 (N-terminal) and ACE (C-terminal) specificity.
    """
    n_term = np.zeros(20)
    c_term = np.zeros(20)

    if len(sequence) >= 1:
        idx = STANDARD_AAS.find(sequence[0])
        if idx >= 0:
            n_term[idx] = 1.0

    if len(sequence) >= 1:
        idx = STANDARD_AAS.find(sequence[-1])
        if idx >= 0:
            c_term[idx] = 1.0

    return np.concatenate([n_term, c_term])


def compute_positional_features(sequence: str) -> np.ndarray:
    """Position-specific features (20 features).

    Encodes the second residue (position 2) as one-hot.
    Critical for DPP4 which cleaves X-Pro/X-Ala at position 2.
    """
    p2 = np.zeros(20)
    if len(sequence) >= 2:
        idx = STANDARD_AAS.find(sequence[1])
        if idx >= 0:
            p2[idx] = 1.0
    return p2


def compute_dpp4_motif_features(sequence: str) -> np.ndarray:
    """DPP4-specific structural motifs (8 features).

    DPP4 cleaves after the second residue if position 2 is Pro or Ala.
    These features capture known SAR for DPP4-inhibitory peptides.
    """
    length = len(sequence)
    features = [
        1.0 if length >= 2 and sequence[1] == "P" else 0.0,  # Pro at P2
        1.0 if length >= 2 and sequence[1] == "A" else 0.0,  # Ala at P2
        1.0 if length >= 2 and sequence[1] in "PA" else 0.0,  # Pro or Ala at P2
        1.0 if length >= 3 and sequence[2] in "GILV" else 0.0,  # Hydrophobic at P3
        1.0 if sequence[0] in "IVLF" else 0.0,  # Hydrophobic N-terminus
        1.0 if sequence[0] in "WFY" else 0.0,  # Aromatic N-terminus
        sequence.count("P") / max(length, 1),  # Proline content
        1.0 if "GP" in sequence or "PP" in sequence else 0.0,  # Collagen-like motifs
    ]
    return np.array(features, dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Featurization Function
# ═══════════════════════════════════════════════════════════════════════════════

def featurize_peptide(sequence: str, target: str = "dpp4") -> Optional[np.ndarray]:
    """Convert a peptide sequence to a feature vector for ML prediction.

    Args:
        sequence: Amino acid sequence (uppercase 1-letter codes).
        target: Target enzyme (affects which motif features to include).

    Returns:
        Feature vector (numpy array) or None if invalid sequence.
    """
    # Validate
    sequence = sequence.upper().strip()
    if not sequence or not all(aa in STANDARD_AAS for aa in sequence):
        return None

    # Compute all feature blocks
    aac = compute_aac(sequence)                    # 20
    dpc = compute_dpc(sequence)                    # 400
    physico = compute_physicochemical(sequence)    # 12
    terminal = compute_terminal_features(sequence)  # 40
    positional = compute_positional_features(sequence)  # 20
    motif = compute_dpp4_motif_features(sequence)  # 8

    # Concatenate all features
    feature_vector = np.concatenate([aac, dpc, physico, terminal, positional, motif])
    return feature_vector


def get_feature_names(target: str = "dpp4") -> list[str]:
    """Get feature names for interpretability."""
    names = []

    # AAC (20)
    names.extend([f"aac_{aa}" for aa in STANDARD_AAS])

    # DPC (400)
    for aa1 in STANDARD_AAS:
        for aa2 in STANDARD_AAS:
            names.append(f"dpc_{aa1}{aa2}")

    # Physicochemical (12)
    names.extend([
        "length", "mw", "mean_hydrophobicity", "hydro_std",
        "net_charge", "aromatic_frac", "aliphatic_frac",
        "mean_bulk", "mean_flexibility", "proline_frac",
        "glycine_frac", "hydro_range",
    ])

    # Terminal (40)
    names.extend([f"nterm_{aa}" for aa in STANDARD_AAS])
    names.extend([f"cterm_{aa}" for aa in STANDARD_AAS])

    # Positional (20)
    names.extend([f"pos2_{aa}" for aa in STANDARD_AAS])

    # Motif (8)
    names.extend([
        "pro_at_p2", "ala_at_p2", "pro_or_ala_p2",
        "hydrophobic_p3", "hydrophobic_nterm", "aromatic_nterm",
        "proline_content", "collagen_motif",
    ])

    return names


def featurize_dataset(sequences: list[str], target: str = "dpp4") -> Optional[np.ndarray]:
    """Featurize a list of peptide sequences into a feature matrix.

    Args:
        sequences: List of peptide sequences.
        target: Target enzyme.

    Returns:
        2D numpy array (n_samples x n_features) or None if all fail.
    """
    features = []
    valid_indices = []
    for i, seq in enumerate(sequences):
        fv = featurize_peptide(seq, target)
        if fv is not None:
            features.append(fv)
            valid_indices.append(i)

    if not features:
        return None

    return np.array(features), valid_indices


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Dimensionality
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_DIM = 500  # 20 + 400 + 12 + 40 + 20 + 8
