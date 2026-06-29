"""
Molecular Docking Pipeline — AutoDock Vina Wrapper.

Provides docking functionality for peptide-target scoring.
Requires AutoDock Vina binary to be installed separately.

Download Vina from: https://github.com/ccsb-scripps/AutoDock-Vina/releases
Place the binary in the project root or add to PATH.
"""

import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

# Project root directory (parent of training/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Docking box configurations for each target (from co-crystallized ligands in PDB)
# center_x, center_y, center_z, size_x, size_y, size_z
DOCKING_CONFIGS = {
    "ace": {
        "pdb_id": "1O86",
        "receptor_file": "receptors/ace_receptor.pdbqt",
        "center": (41.2, 34.5, 45.4),
        "box_size": (28.0, 27.0, 27.0),
        "description": "ACE active site (Zn2+ binding region, lisinopril pocket)",
    },
    "dpp4": {
        "pdb_id": "1X70",
        "receptor_file": "receptors/dpp4_receptor.pdbqt",
        "center": (44.2, 57.5, 36.6),
        "box_size": (30.0, 30.0, 30.0),
        "description": "DPP-IV catalytic domain (S1/S2 subsites)",
    },
    "amylase": {
        "pdb_id": "1B2Y",
        "receptor_file": "receptors/amylase_receptor.pdbqt",
        "center": (13.9, 20.0, 65.0),
        "box_size": (22.0, 22.0, 29.0),
        "description": "Alpha-amylase active site cleft",
    },
    "glucosidase": {
        "pdb_id": "3TOP",
        "receptor_file": "receptors/glucosidase_receptor.pdbqt",
        "center": (-15.0, -5.0, 20.0),
        "box_size": (25.0, 25.0, 25.0),
        "description": "Alpha-glucosidase catalytic domain",
    },
    "lipase": {
        "pdb_id": "1LPB",
        "receptor_file": "receptors/lipase_receptor.pdbqt",
        "center": (1.8, 10.7, 26.1),
        "box_size": (25.0, 29.0, 22.0),
        "description": "Pancreatic lipase active site (Ser-His-Asp triad)",
    },
    "egfr": {
        "pdb_id": "1M17",
        "receptor_file": "receptors/egfr_receptor.pdbqt",
        "center": (22.0, 0.3, 52.8),
        "box_size": (26.0, 22.0, 22.0),
        "description": "EGFR kinase ATP-binding site (erlotinib pocket)",
    },
    "her2": {
        "pdb_id": "3PP0",
        "receptor_file": "receptors/her2_receptor.pdbqt",
        "center": (17.1, 16.5, 26.6),
        "box_size": (22.0, 24.0, 22.0),
        "description": "HER2 kinase domain",
    },
    "vegfr2": {
        "pdb_id": "4ASD",
        "receptor_file": "receptors/vegfr2_receptor.pdbqt",
        "center": (-24.6, -0.4, -10.9),
        "box_size": (23.0, 22.0, 22.0),
        "description": "VEGFR2 kinase ATP-binding site",
    },
    "braf": {
        "pdb_id": "4MNE",
        "receptor_file": "receptors/braf_receptor.pdbqt",
        "center": (-3.5, -13.1, -43.4),
        "box_size": (25.0, 27.0, 22.0),
        "description": "BRAF V600E kinase domain",
    },
    "cdk2": {
        "pdb_id": "1HCK",
        "receptor_file": "receptors/cdk2_receptor.pdbqt",
        "center": (100.5, 97.9, 81.7),
        "box_size": (22.0, 22.0, 22.0),
        "description": "CDK2 ATP-binding site",
    },
}

# Vina binary locations to search
VINA_SEARCH_PATHS = [
    str(_PROJECT_ROOT / "tools" / "vina_1.2.5_win.exe"),
    str(_PROJECT_ROOT / "tools" / "vina.exe"),
    "vina",
    "vina.exe",
    "vina_1.2.5_win.exe",
    r"C:\Program Files\Vina\vina.exe",
]


def find_vina_binary() -> str | None:
    """Find the AutoDock Vina binary."""
    import shutil
    for path in VINA_SEARCH_PATHS:
        if shutil.which(path):
            return shutil.which(path)
        if Path(path).exists():
            return str(Path(path).resolve())
    return None


def is_docking_available() -> bool:
    """Check if docking infrastructure is available."""
    return find_vina_binary() is not None


def smiles_to_pdbqt(smiles: str, output_path: str) -> bool:
    """Convert SMILES to PDBQT format for docking.

    Uses RDKit for 3D coordinate generation and meeko for PDBQT conversion.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)

        # Try ETKDGv3, fall back to simpler methods for large molecules
        params = AllChem.ETKDGv3()
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            # Fallback: use random coordinates
            params.useRandomCoords = True
            result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            # Last resort: basic distance geometry
            result = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
        if result == -1:
            return False

        # Minimize energy
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            pass  # Proceed without optimization if it fails

        # Try meeko for PDBQT conversion
        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(mol_setups[0])
            if pdbqt_string:
                with open(output_path, "w") as f:
                    f.write(pdbqt_string)
                return True
            else:
                logger.error(f"meeko PDBQT write failed: {err}")
                return False
        except (ImportError, Exception) as e:
            logger.warning(f"meeko failed ({e}), falling back to manual PDBQT")
            # Fallback: manual PDBQT from RDKit PDB
            pdb_block = Chem.MolToPDBBlock(mol)
            if not pdb_block:
                return False
            pdbqt_lines = []
            for line in pdb_block.splitlines():
                if line.startswith("HETATM") or line.startswith("ATOM"):
                    atom_name = line[12:16].strip()
                    element = line[76:78].strip() if len(line) >= 78 else atom_name[0]
                    ad_type = element
                    if element == "O" or element == "N" or element == "S":
                        ad_type = element + "A" if element != "N" else "NA"
                    pdbqt_line = f"{line[:54].ljust(54)}{line[54:66] if len(line) >= 66 else '  1.00  0.00'}    +0.000 {ad_type:<2}\n"
                    pdbqt_lines.append(pdbqt_line)
            with open(output_path, "w") as f:
                f.writelines(pdbqt_lines)
            return True

    except Exception as e:
        logger.error(f"Failed to prepare ligand: {e}")
        return False


def run_docking(smiles: str, target: str, exhaustiveness: int = 8,
                num_modes: int = 5, timeout_sec: int = 180) -> dict | None:
    """Run AutoDock Vina docking for a compound against a target.

    This is a pure execution function. Feasibility assessment should be
    performed BEFORE calling this function (see docking_feasibility.py).

    Args:
        smiles: SMILES string of the ligand.
        target: Target name (lowercase).
        exhaustiveness: Vina exhaustiveness parameter (default 8).
        num_modes: Number of binding modes to generate (default 5).
        timeout_sec: Maximum time in seconds for Vina execution (default 180).

    Returns:
        Dictionary with docking results or None if docking unavailable/failed.
    """
    from rdkit.Chem import rdMolDescriptors

    vina_binary = find_vina_binary()
    if vina_binary is None:
        return {"error": "AutoDock Vina not found. Install from: "
                "https://github.com/ccsb-scripps/AutoDock-Vina/releases"}

    target_key = target.lower()
    config = DOCKING_CONFIGS.get(target_key)
    if config is None:
        return {"error": f"No docking configuration for target: {target}"}

    receptor_path = _PROJECT_ROOT / config["receptor_file"]
    if not receptor_path.exists():
        return {"error": f"Receptor file not found: {receptor_path}. "
                f"Download PDB {config['pdb_id']} and prepare with prepare_receptor."}

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"error": "Invalid SMILES string"}

    n_rot = rdMolDescriptors.CalcNumRotatableBonds(Chem.AddHs(mol))

    with tempfile.TemporaryDirectory() as tmpdir:
        ligand_path = str(Path(tmpdir) / "ligand.pdbqt")
        output_path = str(Path(tmpdir) / "output.pdbqt")

        # Prepare ligand
        if not smiles_to_pdbqt(smiles, ligand_path):
            return {"error": "Failed to prepare ligand for docking"}

        # Run Vina 1.2.5 (no --log flag, results in stdout)
        cx, cy, cz = config["center"]
        sx, sy, sz = config["box_size"]

        cmd = [
            vina_binary,
            "--receptor", str(receptor_path),
            "--ligand", ligand_path,
            "--out", output_path,
            "--center_x", str(cx),
            "--center_y", str(cy),
            "--center_z", str(cz),
            "--size_x", str(sx),
            "--size_y", str(sy),
            "--size_z", str(sz),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(num_modes),
        ]

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_sec
            )
        except subprocess.TimeoutExpired:
            return {"error": f"Docking timed out (>{timeout_sec}s). Molecule has "
                    f"{n_rot} rotatable bonds — too complex for exhaustive search."}
        except FileNotFoundError:
            return {"error": "Vina binary not executable"}

        if proc.returncode != 0:
            return {"error": f"Vina failed: {proc.stderr[:200]}"}

        # Parse results from stdout (Vina 1.2.5 format)
        scores = parse_vina_output(proc.stdout)
        if not scores:
            return {"error": "Failed to parse docking results"}

        return {
            "target": target_key,
            "smiles": smiles,
            "best_affinity_kcal": scores[0],
            "all_affinities": scores,
            "n_rotatable_bonds": n_rot,
            "exhaustiveness_used": exhaustiveness,
            "pdb_id": config["pdb_id"],
            "description": config["description"],
        }


def parse_vina_output(stdout: str) -> list[float]:
    """Parse binding affinities from Vina 1.2.5 stdout output."""
    scores = []
    in_results = False
    for line in stdout.splitlines():
        if "-----+--------" in line:
            in_results = True
            continue
        if in_results:
            line = line.strip()
            if not line:
                break
            parts = line.split()
            if len(parts) >= 2:
                try:
                    scores.append(float(parts[1]))
                except ValueError:
                    continue
    return scores


def normalize_docking_score(affinity_kcal: float, target: str) -> float:
    """Normalize docking affinity to 0-1 scale.

    Uses target-specific ranges based on typical binding energies.
    More negative = better binding = higher score.

    Args:
        affinity_kcal: Binding affinity in kcal/mol (negative = good).
        target: Target name.

    Returns:
        Normalized score between 0 and 1.
    """
    # Typical ranges: strong binder = -12 kcal/mol, non-binder = -3 kcal/mol
    min_good = -12.0  # Best possible
    max_bad = -3.0    # Worst (essentially no binding)

    # Clamp and normalize
    clamped = max(min_good, min(max_bad, affinity_kcal))
    normalized = (max_bad - clamped) / (max_bad - min_good)
    return float(np.clip(normalized, 0.0, 1.0))


def compute_composite_score(
    compatibility_score: float,
    docking_score: float | None,
    weight_physico: float = 0.4,
    weight_docking: float = 0.6,
) -> dict:
    """Compute weighted composite score from physicochemical and docking components.

    Args:
        compatibility_score: Physicochemical compatibility (0-1).
        docking_score: Normalized docking score (0-1) or None if unavailable.
        weight_physico: Weight for physicochemical component.
        weight_docking: Weight for docking component.

    Returns:
        Dictionary with composite score and breakdown.
    """
    if docking_score is not None:
        composite = (weight_physico * compatibility_score) + (weight_docking * docking_score)
        method = "composite (physicochemical + docking)"
    else:
        composite = compatibility_score
        method = "physicochemical only (docking unavailable)"

    return {
        "composite_score": float(composite),
        "compatibility_score": float(compatibility_score),
        "docking_score": docking_score,
        "method": method,
        "weights": {"physicochemical": weight_physico, "docking": weight_docking},
    }



def _run_vina_only(
    vina_binary: str,
    receptor_path: str,
    ligand_path: str,
    output_path: str,
    config: dict,
    exhaustiveness: int,
    num_modes: int,
    timeout_sec: int,
) -> dict:
    """Run Vina subprocess only (no RDKit calls — thread-safe)."""
    cx, cy, cz = config["center"]
    sx, sy, sz = config["box_size"]

    cmd = [
        vina_binary,
        "--receptor", receptor_path,
        "--ligand", ligand_path,
        "--out", output_path,
        "--center_x", str(cx),
        "--center_y", str(cy),
        "--center_z", str(cz),
        "--size_x", str(sx),
        "--size_y", str(sy),
        "--size_z", str(sz),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_sec
        )
    except subprocess.TimeoutExpired:
        return {"error": f"Docking timed out (>{timeout_sec}s)"}
    except FileNotFoundError:
        return {"error": "Vina binary not executable"}

    if proc.returncode != 0:
        return {"error": f"Vina failed: {proc.stderr[:200]}"}

    scores = parse_vina_output(proc.stdout)
    if not scores:
        return {"error": "Failed to parse docking results"}

    return {"best_affinity_kcal": scores[0], "all_affinities": scores}


def run_batch_docking(
    jobs: list[tuple[str, str]],
    exhaustiveness: int = 8,
    max_workers: int | None = None,
    progress_callback=None,
) -> list[dict]:
    """Run multiple docking jobs with parallel Vina execution.

    Ligand preparation (RDKit/meeko) runs sequentially in the main thread
    because RDKit is not thread-safe. Vina subprocesses run in parallel.

    Args:
        jobs: List of (smiles, target_key) tuples.
        exhaustiveness: Vina exhaustiveness parameter.
        max_workers: Number of parallel Vina processes. Defaults to min(cpu_count/2, len(jobs), 4).
        progress_callback: Optional callable(completed, total) for progress updates.

    Returns:
        List of docking result dicts, in the same order as input jobs.
    """
    from rdkit.Chem import rdMolDescriptors

    if not jobs:
        return []

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(jobs), 8)
    max_workers = max(1, max_workers)

    vina_binary = find_vina_binary()
    if vina_binary is None:
        return [{"error": "AutoDock Vina not found"}] * len(jobs)

    results = [None] * len(jobs)
    # Temp directory persists across all jobs in this batch
    import tempfile as _tmpmod
    batch_tmpdir = _tmpmod.mkdtemp(prefix="vina_batch_")

    # Phase A: Prepare all ligands sequentially (RDKit not thread-safe)
    prepared_jobs = []  # (index, ligand_path, output_path, config, exh, num_modes, timeout, metadata)
    for i, (smiles, target) in enumerate(jobs):
        target_key = target.lower()
        config = DOCKING_CONFIGS.get(target_key)
        if config is None:
            results[i] = {"error": f"No docking configuration for target: {target}"}
            continue

        receptor_path = _PROJECT_ROOT / config["receptor_file"]
        if not receptor_path.exists():
            results[i] = {"error": f"Receptor file not found: {receptor_path}"}
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results[i] = {"error": "Invalid SMILES string"}
            continue

        n_rot = rdMolDescriptors.CalcNumRotatableBonds(Chem.AddHs(mol))

        # Adaptive parameters based on complexity
        if n_rot > 25:
            exh = min(exhaustiveness, 1)
            nm = 3
            timeout = 300
        elif n_rot > 15:
            exh = min(exhaustiveness, 4)
            nm = 3
            timeout = 240
        else:
            exh = exhaustiveness
            nm = 5
            timeout = 180

        ligand_path = str(Path(batch_tmpdir) / f"ligand_{i}.pdbqt")
        output_path = str(Path(batch_tmpdir) / f"output_{i}.pdbqt")

        if not smiles_to_pdbqt(smiles, ligand_path):
            results[i] = {"error": "Failed to prepare ligand for docking"}
            continue

        prepared_jobs.append((i, ligand_path, output_path, config,
                             str(receptor_path), exh, nm, timeout,
                             {"target": target_key, "smiles": smiles,
                              "n_rotatable_bonds": n_rot, "pdb_id": config["pdb_id"],
                              "description": config["description"]}))

    # Phase B: Run Vina in parallel (subprocess only, thread-safe)
    total = len(jobs)
    completed = total - len(prepared_jobs)  # Pre-failed jobs count as done

    if prepared_jobs:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for (idx, lig_path, out_path, cfg, rec_path, exh, nm, timeout, meta) in prepared_jobs:
                future = executor.submit(
                    _run_vina_only, vina_binary, rec_path, lig_path, out_path,
                    cfg, exh, nm, timeout
                )
                futures[future] = (idx, meta, exh)

            for future in as_completed(futures):
                idx, meta, exh_used = futures[future]
                try:
                    vina_result = future.result()
                    if "error" in vina_result:
                        results[idx] = vina_result
                    else:
                        results[idx] = {
                            **meta,
                            **vina_result,
                            "exhaustiveness_used": exh_used,
                        }
                except Exception as e:
                    logger.error(f"Docking job {idx} exception: {e}")
                    results[idx] = {"error": str(e)}
                completed += 1
                if progress_callback:
                    try:
                        progress_callback(completed, total)
                    except Exception:
                        pass

    # Cleanup temp dir
    import shutil
    shutil.rmtree(batch_tmpdir, ignore_errors=True)

    return results
