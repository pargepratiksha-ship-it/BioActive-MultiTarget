"""
Receptor Preparation Script.

Downloads PDB structures and converts to PDBQT format for AutoDock Vina.
Uses RDKit + biopython for structure processing and Open Babel-free approach.

For each target:
1. Download PDB file from RCSB
2. Remove water, ligands, alternate conformations
3. Keep only protein chain A (or relevant chain)
4. Write as PDBQT (simplified — adds Gasteiger charges, assigns AD atom types)
"""

import urllib.request
from pathlib import Path

from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.PDBIO import PDBIO

# Target → PDB ID mapping (same as docking_pipeline.py)
TARGET_PDBS = {
    "ace": {"pdb_id": "1O86", "chain": "A", "description": "ACE with lisinopril"},
    "dpp4": {"pdb_id": "1X70", "chain": "A", "description": "DPP-IV with sitagliptin-like"},
    "amylase": {"pdb_id": "1B2Y", "chain": "A", "description": "Human pancreatic amylase"},
    "glucosidase": {"pdb_id": "3TOP", "chain": "A", "description": "Alpha-glucosidase"},
    "lipase": {"pdb_id": "1LPB", "chain": "A", "description": "Human pancreatic lipase"},
    "egfr": {"pdb_id": "1M17", "chain": "A", "description": "EGFR kinase with erlotinib"},
    "her2": {"pdb_id": "3PP0", "chain": "A", "description": "HER2 kinase domain"},
    "vegfr2": {"pdb_id": "4ASD", "chain": "A", "description": "VEGFR2 kinase"},
    "braf": {"pdb_id": "4MNE", "chain": "A", "description": "BRAF V600E with inhibitor"},
    "cdk2": {"pdb_id": "1HCK", "chain": "A", "description": "CDK2 with ATP"},
}

RECEPTORS_DIR = Path("receptors")
PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# Atom type mapping for PDBQT (simplified AutoDock atom types)
AD_ATOM_TYPES = {
    "C": "C", "CA": "C", "CB": "C", "CG": "C", "CD": "C", "CE": "C", "CZ": "C",
    "CG1": "C", "CG2": "C", "CD1": "C", "CD2": "C", "CE1": "C", "CE2": "C",
    "CH2": "C",
    "N": "N", "NE": "N", "NE1": "N", "NE2": "N", "ND1": "N", "ND2": "N",
    "NH1": "N", "NH2": "N", "NZ": "N",
    "O": "OA", "OG": "OA", "OG1": "OA", "OD1": "OA", "OD2": "OA",
    "OE1": "OA", "OE2": "OA", "OH": "OA", "OXT": "OA",
    "S": "SA", "SD": "SA", "SG": "SA",
    "H": "HD",
}


class ProteinSelect(Select):
    """Select only protein atoms from specified chain, no water/hetero."""

    def __init__(self, chain_id: str):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return chain.get_id() == self.chain_id

    def accept_residue(self, residue):
        # Reject water and hetero atoms (except MSE → MET)
        hetfield = residue.get_id()[0]
        return hetfield == " " or hetfield == "H_MSE"

    def accept_atom(self, atom):
        # Reject alternate conformations (keep 'A' or ' ')
        altloc = atom.get_altloc()
        return altloc == " " or altloc == "A"


def download_pdb(pdb_id: str, output_path: Path) -> bool:
    """Download PDB file from RCSB."""
    url = PDB_DOWNLOAD_URL.format(pdb_id=pdb_id)
    try:
        urllib.request.urlretrieve(url, str(output_path))
        return True
    except Exception as e:
        print(f"  ERROR downloading {pdb_id}: {e}")
        return False


def pdb_to_pdbqt(pdb_path: Path, pdbqt_path: Path) -> bool:
    """Convert cleaned PDB to PDBQT format.

    Simplified conversion: assigns AD4 atom types and zero partial charges.
    For production use, prepare_receptor4.py from ADFRsuite is preferred.
    """
    try:
        lines = []
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    atom_name = line[12:16].strip()
                    element = line[76:78].strip() if len(line) >= 78 else atom_name[0]

                    # Determine AD atom type
                    ad_type = AD_ATOM_TYPES.get(atom_name, element)
                    if not ad_type:
                        ad_type = element if element else "C"

                    # PDBQT format: PDB + partial charge (col 71-76) + atom type (col 77-79)
                    # Pad line to 54 chars, add occupancy/bfactor columns then charge + type
                    pdb_line = line[:54].ljust(54)
                    # Keep occupancy and b-factor from original
                    occ_bfac = line[54:66] if len(line) >= 66 else "  1.00  0.00"
                    # Partial charge (0.000 placeholder) + atom type
                    pdbqt_line = f"{pdb_line}{occ_bfac}    +0.000 {ad_type:<2}\n"
                    lines.append(pdbqt_line)

        with open(pdbqt_path, "w") as f:
            f.writelines(lines)
        return True
    except Exception as e:
        print(f"  ERROR converting to PDBQT: {e}")
        return False


def prepare_receptor(target: str) -> bool:
    """Download and prepare receptor PDBQT for a target."""
    config = TARGET_PDBS.get(target)
    if config is None:
        print(f"  Unknown target: {target}")
        return False

    pdb_id = config["pdb_id"]
    chain = config["chain"]

    print(f"\n[{target.upper()}] PDB: {pdb_id}, Chain: {chain}")
    print(f"  {config['description']}")

    # Download
    raw_pdb = RECEPTORS_DIR / f"{pdb_id}_raw.pdb"
    if not raw_pdb.exists():
        print(f"  Downloading {pdb_id}...")
        if not download_pdb(pdb_id, raw_pdb):
            return False
        print(f"  Downloaded: {raw_pdb}")
    else:
        print(f"  Already downloaded: {raw_pdb}")

    # Parse and clean
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(raw_pdb))

    clean_pdb = RECEPTORS_DIR / f"{target}_receptor_clean.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(clean_pdb), ProteinSelect(chain))
    print(f"  Cleaned PDB saved: {clean_pdb}")

    # Convert to PDBQT
    pdbqt_path = RECEPTORS_DIR / f"{target}_receptor.pdbqt"
    if pdb_to_pdbqt(clean_pdb, pdbqt_path):
        # Count atoms
        n_atoms = sum(1 for line in open(pdbqt_path) if line.startswith("ATOM"))
        print(f"  PDBQT saved: {pdbqt_path} ({n_atoms} atoms)")
        return True
    return False


def prepare_all_receptors():
    """Prepare all 10 receptor PDBQT files."""
    print("=" * 60)
    print("Receptor Preparation — AutoDock Vina")
    print("=" * 60)

    RECEPTORS_DIR.mkdir(exist_ok=True)

    success = 0
    failed = []
    for target in TARGET_PDBS:
        if prepare_receptor(target):
            success += 1
        else:
            failed.append(target)

    print(f"\n{'=' * 60}")
    print(f"Results: {success}/{len(TARGET_PDBS)} receptors prepared")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    else:
        print("All receptors ready for docking!")


if __name__ == "__main__":
    prepare_all_receptors()
