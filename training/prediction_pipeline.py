"""
Unified Prediction Pipeline.

Standardizes the prediction workflow for ALL ligand types (small molecules
and peptides) through an identical orchestration sequence:

    Input → Validation → ML Prediction → Physicochemical Analysis →
    Docking Feasibility → [Docking] → Evidence Integration →
    Confidence → Final Prediction

The pipeline ensures:
- Every ligand follows the same stages regardless of type
- Every prediction carries an explicit evaluation status
- Docking is only attempted when scientifically appropriate
- Transparent reporting of what was evaluated and what was skipped
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity

from training.docking_feasibility import (
    FeasibilityRecommendation,
    FeasibilityResult,
    FeasibilityThresholds,
    assess_docking_feasibility,
    get_adaptive_docking_parameters,
)
from training.docking_pipeline import (
    DOCKING_CONFIGS,
    is_docking_available,
    normalize_docking_score,
    run_docking,
)
from training.peptide_scoring import compute_compatibility_score

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
AD_SIMILARITY_THRESHOLD = 0.3

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "processed"
PEPTIDE_MODEL_DIR = _PROJECT_ROOT / "models"


# ═══════════════════════════════════════════════════════════════════════════════
# Enums and Result Types
# ═══════════════════════════════════════════════════════════════════════════════

class EvaluationStatus(str, Enum):
    """Classification of prediction completeness."""

    FULLY_EVALUATED = "fully_evaluated"
    """All applicable evidence layers completed successfully (ML + Physico + Docking)."""

    PARTIALLY_EVALUATED = "partially_evaluated"
    """Some evidence layers completed, but docking was skipped due to structural
    limitations. The prediction is still informative but carries lower confidence."""

    INVALID_INPUT = "invalid_input"
    """The input could not be parsed as a valid molecule or peptide sequence."""

    FAILED_PROCESSING = "failed_processing"
    """An unexpected error occurred during computation."""


class LigandType(str, Enum):
    """Type of input ligand."""
    SMALL_MOLECULE = "small_molecule"
    PEPTIDE = "peptide"


@dataclass
class StageResult:
    """Result from a single pipeline stage."""
    status: str  # "completed", "skipped", "failed", "not_applicable"
    score: float | None = None
    details: dict = field(default_factory=dict)
    error: str | None = None


@dataclass
class PredictionResult:
    """Complete prediction result with full transparency.

    Every field documents what was computed, what was skipped, and why.
    """
    # ── Input ──
    input_text: str
    ligand_type: LigandType
    smiles: str | None = None
    peptide_sequence: str | None = None
    target: str = ""

    # ── Evaluation Status ──
    evaluation_status: EvaluationStatus = EvaluationStatus.FAILED_PROCESSING

    # ── Stage Results ──
    validation: StageResult = field(default_factory=lambda: StageResult(status="not_run"))
    ml_prediction: StageResult = field(default_factory=lambda: StageResult(status="not_run"))
    physicochemical: StageResult = field(default_factory=lambda: StageResult(status="not_run"))
    docking_feasibility: StageResult = field(default_factory=lambda: StageResult(status="not_run"))
    docking: StageResult = field(default_factory=lambda: StageResult(status="not_run"))

    # ── Integrated Scores ──
    final_score: float | None = None
    prediction_confidence: float | None = None
    primary_method: str = ""

    # ── Flags ──
    flag: str | None = None
    flag_reason: str | None = None

    # ── Convenience Properties ──
    @property
    def ml_score(self) -> float | None:
        return self.ml_prediction.score

    @property
    def physico_score(self) -> float | None:
        return self.physicochemical.score

    @property
    def docking_score(self) -> float | None:
        return self.docking.score

    @property
    def docking_affinity(self) -> float | None:
        return self.docking.details.get("affinity_kcal")

    @property
    def docking_skip_reason(self) -> str | None:
        if self.docking.status == "skipped":
            return self.docking.error
        if self.docking_feasibility.status == "completed" and not self.docking_feasibility.details.get("is_suitable"):
            reasons = self.docking_feasibility.details.get("skip_reasons", [])
            return "; ".join(reasons) if reasons else "Docking not feasible"
        return None

    @property
    def bioactivity_percent(self) -> float | None:
        if self.final_score is not None:
            return round(self.final_score * 100, 1)
        return None

    @property
    def ad_score(self) -> float | None:
        return self.ml_prediction.details.get("ad_score")

    @property
    def in_domain(self) -> bool | None:
        return self.ml_prediction.details.get("in_domain")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class PredictionPipeline:
    """Unified prediction pipeline for any ligand type.

    Orchestrates the same sequence of stages for every input:
    Validation → ML → Physicochemical → Feasibility → Docking → Integration

    Args:
        model: Trained ML model with predict_proba method (small-molecule).
        target: Target name (display name, e.g., "DPP4").
        target_key: Internal target key (e.g., "dpp4").
        training_fingerprints: Precomputed training set fingerprints for AD check.
        feasibility_thresholds: Custom thresholds for docking feasibility.
        peptide_model_data: Optional loaded peptide model dict (from joblib).
            If None, will attempt to auto-load from models/{target_key}_peptide_model.joblib.
    """

    def __init__(
        self,
        model,
        target: str,
        target_key: str,
        training_fingerprints: np.ndarray | None = None,
        feasibility_thresholds: FeasibilityThresholds | None = None,
        peptide_model_data: dict | None = None,
    ):
        self.model = model
        self.target = target
        self.target_key = target_key
        self.training_fingerprints = training_fingerprints
        self.feasibility_thresholds = feasibility_thresholds

        # Load peptide model
        self.peptide_model_data = peptide_model_data
        if self.peptide_model_data is None:
            self._try_load_peptide_model()

    def predict(
        self,
        input_text: str,
        run_docking: bool = False,
    ) -> PredictionResult:
        """Run the full prediction pipeline.

        Args:
            input_text: SMILES string or peptide sequence.
            run_docking: Whether to attempt docking (subject to feasibility).

        Returns:
            Complete PredictionResult with all stage outcomes.
        """
        # Detect input type
        ligand_type = self._detect_ligand_type(input_text)

        result = PredictionResult(
            input_text=input_text,
            ligand_type=ligand_type,
            target=self.target,
        )

        # ── Stage 1: Validation ──
        self._run_validation(result)
        if result.evaluation_status == EvaluationStatus.INVALID_INPUT:
            return result

        # ── Stage 2: ML Prediction ──
        self._run_ml_prediction(result)

        # ── Stage 3: Physicochemical Analysis ──
        self._run_physicochemical(result)

        # ── Stage 4: Docking Feasibility Assessment ──
        self._run_docking_feasibility(result)

        # ── Stage 5: Molecular Docking (conditional) ──
        if run_docking:
            self._run_docking(result)
        else:
            result.docking = StageResult(status="not_requested")

        # ── Stage 6: Evidence Integration ──
        self._integrate_evidence(result)

        # ── Stage 7: Confidence & Status ──
        self._assign_confidence_and_status(result)

        # ── Stage 8: Disagreement Flagging ──
        self._check_disagreement(result)

        return result

    # ───────────────────────────────────────────────────────────────────────────
    # Stage Implementations
    # ───────────────────────────────────────────────────────────────────────────

    def _detect_ligand_type(self, input_text: str) -> LigandType:
        """Detect whether input is a SMILES string or peptide sequence."""
        text = input_text.strip().upper()
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        if text and len(text) >= 2 and all(c in valid_aa for c in text):
            return LigandType.PEPTIDE
        return LigandType.SMALL_MOLECULE

    def _run_validation(self, result: PredictionResult) -> None:
        """Stage 1: Validate and normalize input."""
        input_text = result.input_text.strip()

        if result.ligand_type == LigandType.PEPTIDE:
            sequence = input_text.upper()
            valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
            invalid_chars = set(sequence) - valid_aa
            if invalid_chars:
                result.validation = StageResult(
                    status="failed",
                    error=f"Invalid amino acid characters: {', '.join(sorted(invalid_chars))}",
                )
                result.evaluation_status = EvaluationStatus.INVALID_INPUT
                return

            mol = Chem.MolFromSequence(sequence)
            if mol is None:
                result.validation = StageResult(
                    status="failed",
                    error=f"Failed to generate molecule from sequence '{sequence}'",
                )
                result.evaluation_status = EvaluationStatus.INVALID_INPUT
                return

            smiles = Chem.MolToSmiles(mol)
            if not smiles:
                result.validation = StageResult(
                    status="failed",
                    error="Generated empty SMILES from peptide sequence",
                )
                result.evaluation_status = EvaluationStatus.INVALID_INPUT
                return

            result.smiles = smiles
            result.peptide_sequence = sequence
            result.validation = StageResult(
                status="completed",
                details={"canonical_smiles": smiles, "peptide_length": len(sequence)},
            )

        else:  # SMALL_MOLECULE
            mol = Chem.MolFromSmiles(input_text)
            if mol is None:
                result.validation = StageResult(
                    status="failed",
                    error=f"Invalid SMILES: '{input_text}'",
                )
                result.evaluation_status = EvaluationStatus.INVALID_INPUT
                return

            canonical = Chem.MolToSmiles(mol)
            result.smiles = canonical
            result.validation = StageResult(
                status="completed",
                details={"canonical_smiles": canonical},
            )

    def _run_ml_prediction(self, result: PredictionResult) -> None:
        """Stage 2: ML model prediction with applicability domain check.

        For peptides: uses peptide-specific model (sequence features) if available.
        For small molecules: uses Morgan FP-based model.
        """
        # Try peptide-specific model first for peptides
        if result.ligand_type == LigandType.PEPTIDE:
            if self._run_peptide_ml_prediction(result):
                return  # Peptide model used successfully

        # Fallback: Morgan fingerprint-based model
        if result.smiles is None:
            result.ml_prediction = StageResult(status="failed", error="No valid SMILES")
            return

        mol = Chem.MolFromSmiles(result.smiles)
        if mol is None:
            result.ml_prediction = StageResult(status="failed", error="Cannot parse SMILES")
            return

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
        fp_array = np.array(fp)

        try:
            prob = float(self.model.predict_proba(fp_array.reshape(1, -1))[0, 1])
        except Exception as e:
            result.ml_prediction = StageResult(status="failed", error=str(e))
            return

        # Applicability domain check
        ad_score = self._compute_ad_score(fp_array)
        in_domain = ad_score >= AD_SIMILARITY_THRESHOLD if ad_score is not None else None

        result.ml_prediction = StageResult(
            status="completed",
            score=prob,
            details={
                "prediction": "Active" if prob >= 0.5 else "Inactive",
                "model_type": "small_molecule" if result.ligand_type == LigandType.SMALL_MOLECULE else "small_molecule_fallback",
                "ad_score": ad_score,
                "in_domain": in_domain,
            },
        )

    def _run_physicochemical(self, result: PredictionResult) -> None:
        """Stage 3: Physicochemical compatibility scoring."""
        if result.ligand_type == LigandType.PEPTIDE and result.peptide_sequence:
            try:
                physico_result = compute_compatibility_score(
                    result.peptide_sequence, self.target_key
                )
                if physico_result:
                    result.physicochemical = StageResult(
                        status="completed",
                        score=physico_result["compatibility_score"],
                        details={"components": physico_result.get("component_scores", {})},
                    )
                else:
                    result.physicochemical = StageResult(
                        status="failed",
                        error="No physicochemical profile for this target",
                    )
            except Exception as e:
                result.physicochemical = StageResult(status="failed", error=str(e))
        else:
            # Small molecules: physicochemical not applicable in current implementation
            result.physicochemical = StageResult(status="not_applicable")

    def _run_docking_feasibility(self, result: PredictionResult) -> None:
        """Stage 4: Assess whether docking is scientifically appropriate."""
        if result.smiles is None:
            result.docking_feasibility = StageResult(status="failed", error="No valid SMILES")
            return

        if not is_docking_available():
            result.docking_feasibility = StageResult(
                status="skipped",
                error="Docking infrastructure not available (Vina binary not found)",
            )
            return

        if self.target_key not in DOCKING_CONFIGS:
            result.docking_feasibility = StageResult(
                status="skipped",
                error=f"No docking configuration for target: {self.target_key}",
            )
            return

        peptide_length = len(result.peptide_sequence) if result.peptide_sequence else None

        feasibility = assess_docking_feasibility(
            smiles=result.smiles,
            is_peptide=(result.ligand_type == LigandType.PEPTIDE),
            peptide_length=peptide_length,
            thresholds=self.feasibility_thresholds,
        )

        result.docking_feasibility = StageResult(
            status="completed",
            details={
                "is_suitable": feasibility.is_suitable,
                "recommendation": feasibility.recommendation.value,
                "metrics": feasibility.metrics,
                "skip_reasons": feasibility.skip_reasons,
                "caution_reasons": feasibility.caution_reasons,
                "confidence": feasibility.confidence,
            },
        )

    def _run_docking(self, result: PredictionResult) -> None:
        """Stage 5: Execute molecular docking (only if feasible)."""
        # Check feasibility gate
        feasibility_details = result.docking_feasibility.details
        if not feasibility_details.get("is_suitable", False):
            skip_reasons = feasibility_details.get("skip_reasons", ["Not feasible"])
            result.docking = StageResult(
                status="skipped",
                error="; ".join(skip_reasons) if skip_reasons else "Docking not feasible",
            )
            return

        # Get adaptive parameters from feasibility assessment
        feasibility_obj = FeasibilityResult(
            is_suitable=feasibility_details.get("is_suitable", True),
            recommendation=FeasibilityRecommendation(feasibility_details.get("recommendation", "proceed")),
            metrics=feasibility_details.get("metrics", {}),
            confidence=feasibility_details.get("confidence", 1.0),
        )
        adaptive_params = get_adaptive_docking_parameters(feasibility_obj)

        # Execute docking with adaptive parameters
        docking_result = run_docking(
            result.smiles, self.target_key,
            exhaustiveness=adaptive_params["exhaustiveness"],
            num_modes=adaptive_params["num_modes"],
            timeout_sec=adaptive_params["timeout_sec"],
        )
        if docking_result and "error" not in docking_result:
            affinity = docking_result["best_affinity_kcal"]
            normalized = normalize_docking_score(affinity, self.target_key)
            result.docking = StageResult(
                status="completed",
                score=normalized,
                details={
                    "affinity_kcal": affinity,
                    "all_affinities": docking_result.get("all_affinities", []),
                    "n_rotatable_bonds": docking_result.get("n_rotatable_bonds"),
                    "exhaustiveness_used": docking_result.get("exhaustiveness_used"),
                },
            )
        elif docking_result:
            result.docking = StageResult(
                status="failed",
                error=docking_result.get("error", "Unknown docking error"),
            )
        else:
            result.docking = StageResult(status="failed", error="No result from docking engine")

    def _integrate_evidence(self, result: PredictionResult) -> None:
        """Stage 6: Combine available evidence into a final score."""
        ml_score = result.ml_score
        physico_score = result.physico_score
        docking_score = result.docking_score
        in_domain = result.in_domain
        model_type = result.ml_prediction.details.get("model_type", "")

        # Priority logic:
        # 1. In-domain small molecule (Morgan FP model) → ML is primary
        # 2. Peptide-specific model → ML is primary (trained on peptide data!)
        # 3. ML + Physico + Docking → weighted composite
        # 4. ML + Physico (no docking) → weighted average
        # 5. ML only → ML score

        if in_domain and result.ligand_type == LigandType.SMALL_MOLECULE:
            result.final_score = ml_score
            result.primary_method = "ml"
        elif model_type == "peptide_specific" and ml_score is not None:
            # Peptide-specific model: ML is trained on actual peptide IC50 data
            if physico_score is not None and docking_score is not None:
                # All three: ML 50% + Physico 20% + Docking 30%
                result.final_score = 0.5 * ml_score + 0.2 * physico_score + 0.3 * docking_score
                result.primary_method = "peptide_ml + physicochemical + docking"
            elif physico_score is not None:
                # ML + Physico: ML dominates since it's trained on actual peptide data
                result.final_score = 0.7 * ml_score + 0.3 * physico_score
                result.primary_method = "peptide_ml + physicochemical"
            else:
                result.final_score = ml_score
                result.primary_method = "peptide_ml"
        elif ml_score is not None and physico_score is not None and docking_score is not None:
            # All three available: ML 30% + Physico 30% + Docking 40%
            result.final_score = 0.3 * ml_score + 0.3 * physico_score + 0.4 * docking_score
            result.primary_method = "composite (ml + physicochemical + docking)"
        elif ml_score is not None and physico_score is not None:
            # ML + Physico (docking failed/skipped): ML 40% + Physico 60%
            result.final_score = 0.4 * ml_score + 0.6 * physico_score
            result.primary_method = "composite (ml + physicochemical)"
        elif physico_score is not None:
            result.final_score = physico_score
            result.primary_method = "physicochemical"
        elif ml_score is not None:
            result.final_score = ml_score
            result.primary_method = "ml (outside domain)" if not in_domain else "ml"
        else:
            result.final_score = None
            result.primary_method = "unavailable"

    def _assign_confidence_and_status(self, result: PredictionResult) -> None:
        """Stage 7: Determine evaluation status and prediction confidence."""
        ml_ok = result.ml_prediction.status == "completed"
        physico_ok = result.physicochemical.status == "completed"
        physico_na = result.physicochemical.status == "not_applicable"
        docking_ok = result.docking.status == "completed"
        docking_skipped = result.docking.status in ("skipped", "not_requested")
        docking_failed = result.docking.status == "failed"

        # Determine evaluation status
        if result.ligand_type == LigandType.SMALL_MOLECULE:
            # For small molecules: ML is sufficient. Docking is bonus.
            if ml_ok and (docking_ok or docking_skipped):
                if docking_ok:
                    result.evaluation_status = EvaluationStatus.FULLY_EVALUATED
                elif result.docking.status == "not_requested":
                    # Docking wasn't requested — this is still fully evaluated
                    # for the requested scope
                    result.evaluation_status = EvaluationStatus.FULLY_EVALUATED
                else:
                    result.evaluation_status = EvaluationStatus.PARTIALLY_EVALUATED
            elif ml_ok:
                result.evaluation_status = EvaluationStatus.PARTIALLY_EVALUATED
            else:
                result.evaluation_status = EvaluationStatus.FAILED_PROCESSING
        else:
            # For peptides: ML + Physico is baseline. Docking completes it.
            if ml_ok and physico_ok and docking_ok:
                result.evaluation_status = EvaluationStatus.FULLY_EVALUATED
            elif ml_ok and (physico_ok or physico_na):
                if result.docking.status == "not_requested":
                    result.evaluation_status = EvaluationStatus.FULLY_EVALUATED
                else:
                    result.evaluation_status = EvaluationStatus.PARTIALLY_EVALUATED
            elif ml_ok:
                result.evaluation_status = EvaluationStatus.PARTIALLY_EVALUATED
            else:
                result.evaluation_status = EvaluationStatus.FAILED_PROCESSING

        # Compute prediction confidence (0-1)
        confidence_factors = []

        if ml_ok:
            # ML confidence: higher when prediction is decisive (far from 0.5)
            ml_prob = result.ml_score
            ml_decisiveness = abs(ml_prob - 0.5) * 2  # 0 at 0.5, 1 at 0/1
            confidence_factors.append(0.5 + 0.5 * ml_decisiveness)

            # AD bonus/penalty
            if result.in_domain:
                confidence_factors.append(0.9)
            else:
                confidence_factors.append(0.5)

        if physico_ok:
            confidence_factors.append(0.7)

        if docking_ok:
            confidence_factors.append(0.9)
        elif result.docking.status == "skipped":
            confidence_factors.append(0.4)  # Penalty for missing evidence

        if confidence_factors:
            result.prediction_confidence = round(
                sum(confidence_factors) / len(confidence_factors), 3
            )
        else:
            result.prediction_confidence = 0.0

    def _check_disagreement(self, result: PredictionResult) -> None:
        """Stage 8: Flag predictions with conflicting evidence."""
        ml_score = result.ml_score
        physico_score = result.physico_score
        model_type = result.ml_prediction.details.get("model_type", "")

        if ml_score is not None and physico_score is not None:
            diff = abs(ml_score - physico_score)
            if diff >= 0.4:
                if ml_score < 0.3 and physico_score > 0.7:
                    result.flag = "INVESTIGATE"
                    result.flag_reason = (
                        "ML says inactive but physicochemical profile is highly compatible. "
                        "Could be a novel mechanism — recommend docking analysis."
                    )
                elif ml_score > 0.7 and physico_score < 0.3:
                    result.flag = "CAUTION"
                    result.flag_reason = (
                        "ML says active but physicochemical properties don't match "
                        "typical inhibitors. May be a false positive from the ML model."
                    )
        elif (result.ligand_type == LigandType.PEPTIDE
              and model_type == "small_molecule_fallback"
              and result.in_domain is False):
            result.flag = "OUT_OF_DOMAIN"
            result.flag_reason = (
                "No peptide-specific model available for this target. "
                "Using small-molecule model as fallback — prediction less reliable."
            )

    # ───────────────────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────────────────

    def _compute_ad_score(self, query_fp: np.ndarray) -> float | None:
        """Compute max Tanimoto similarity to training data."""
        if self.training_fingerprints is None:
            return None
        dot_products = self.training_fingerprints @ query_fp
        query_bits = query_fp.sum()
        train_bits = self.training_fingerprints.sum(axis=1)
        unions = train_bits + query_bits - dot_products
        similarities = np.divide(
            dot_products, unions,
            out=np.zeros_like(dot_products, dtype=float),
            where=unions != 0,
        )
        return float(similarities.max())

    def _try_load_peptide_model(self) -> None:
        """Attempt to load a target-specific peptide model."""
        import joblib
        model_path = PEPTIDE_MODEL_DIR / f"{self.target_key}_peptide_model.joblib"
        if model_path.exists():
            try:
                self.peptide_model_data = joblib.load(model_path)
                logger.info(f"Loaded peptide model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load peptide model: {e}")
                self.peptide_model_data = None

    def _run_peptide_ml_prediction(self, result: PredictionResult) -> bool:
        """Run peptide-specific ML prediction using sequence features.

        Returns True if peptide model was used successfully, False otherwise.
        """
        if self.peptide_model_data is None:
            return False

        from training.peptide_features import featurize_peptide

        sequence = result.peptide_sequence
        if not sequence:
            return False

        # Featurize using sequence-based descriptors
        feature_vector = featurize_peptide(sequence, self.target_key)
        if feature_vector is None:
            return False

        try:
            pep_model = self.peptide_model_data["model"]
            pep_scaler = self.peptide_model_data["scaler"]
            kept_features = self.peptide_model_data["kept_features"]

            # Scale and filter features
            fv_scaled = pep_scaler.transform(feature_vector.reshape(1, -1))[:, kept_features]
            prob = float(pep_model.predict_proba(fv_scaled)[0, 1])

            result.ml_prediction = StageResult(
                status="completed",
                score=prob,
                details={
                    "prediction": "Active" if prob >= 0.5 else "Inactive",
                    "model_type": "peptide_specific",
                    "ad_score": None,
                    "in_domain": True,  # Peptide model is always in-domain for peptides
                },
            )
            return True
        except Exception as e:
            logger.warning(f"Peptide model prediction failed: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════════

def run_batch_prediction(
    pipeline: PredictionPipeline,
    inputs: list[str],
    run_docking: bool = False,
    progress_callback=None,
) -> list[PredictionResult]:
    """Run predictions for a batch of inputs.

    Args:
        pipeline: Configured PredictionPipeline instance.
        inputs: List of SMILES strings or peptide sequences.
        run_docking: Whether to attempt docking (subject to feasibility).
        progress_callback: Optional callable(completed, total).

    Returns:
        List of PredictionResults in same order as inputs.
    """
    results = []
    total = len(inputs)
    for i, input_text in enumerate(inputs):
        try:
            r = pipeline.predict(input_text, run_docking=run_docking)
        except Exception as e:
            logger.error(f"Prediction failed for input {i}: {e}")
            r = PredictionResult(
                input_text=input_text,
                ligand_type=LigandType.SMALL_MOLECULE,
                target=pipeline.target,
                evaluation_status=EvaluationStatus.FAILED_PROCESSING,
                validation=StageResult(status="failed", error=str(e)),
            )
        results.append(r)
        if progress_callback:
            try:
                progress_callback(i + 1, total)
            except Exception:
                pass
    return results


def results_to_dataframe(results: list[PredictionResult]) -> "pd.DataFrame":
    """Convert a list of PredictionResults to a pandas DataFrame for display.

    Includes all fields required by the reporting spec:
    ML status, physico status, docking status, skip reason, evaluation status,
    final score, and confidence.
    """
    import pandas as pd

    rows = []
    for r in results:
        rows.append({
            "input": r.input_text,
            "ligand_type": r.ligand_type.value,
            "target": r.target,
            "evaluation_status": r.evaluation_status.value,
            "bioactivity_%": r.bioactivity_percent,
            "final_score": r.final_score,
            "prediction_confidence": r.prediction_confidence,
            "primary_method": r.primary_method,
            "ml_score": r.ml_score,
            "ml_status": r.ml_prediction.status,
            "ad_score": r.ad_score,
            "in_domain": r.in_domain,
            "physico_score": r.physico_score,
            "physico_status": r.physicochemical.status,
            "docking_affinity": r.docking_affinity,
            "docking_normalized": r.docking_score,
            "docking_status": r.docking.status,
            "docking_skip_reason": r.docking_skip_reason,
            "flag": r.flag,
            "flag_reason": r.flag_reason,
        })
    return pd.DataFrame(rows)
