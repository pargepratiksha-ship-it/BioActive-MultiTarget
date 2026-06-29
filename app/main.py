"""
BioActive-MultiTarget — Streamlit Web Application.

Unified prediction pipeline for small molecules and peptides:
  Stage 1: Input Validation
  Stage 2: ML Prediction (Morgan FP → XGBoost)
  Stage 3: Physicochemical Analysis (peptides)
  Stage 4: Docking Feasibility Assessment
  Stage 5: Molecular Docking (if feasible and requested)
  Stage 6: Evidence Integration
  Stage 7: Confidence & Status Assignment

Every prediction carries an explicit EvaluationStatus:
  FULLY_EVALUATED | PARTIALLY_EVALUATED | INVALID_INPUT | FAILED_PROCESSING
"""

import io
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Add project root to path for training module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.docking_pipeline import is_docking_available
from training.prediction_pipeline import (
    EvaluationStatus,
    LigandType,
    PredictionPipeline,
    PredictionResult,
    run_batch_prediction,
    results_to_dataframe,
)

# Configuration — use paths relative to project root, not cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

logger = logging.getLogger(__name__)

# Available targets and their model paths
TARGETS = {
    "DPP4": {
        "model_path": MODEL_DIR / "dpp4_model.joblib",
        "chembl_id": "CHEMBL284",
        "description": "Dipeptidyl Peptidase-4 — diabetes target",
        "activity_threshold": "IC50 ≤ 10 µM",
    },
    "Alpha-Amylase": {
        "model_path": MODEL_DIR / "amylase_model.joblib",
        "chembl_id": "CHEMBL2045",
        "description": "Human Pancreatic Alpha-Amylase — diabetes target",
        "activity_threshold": "IC50 ≤ 10 µM",
    },
    "Alpha-Glucosidase": {
        "model_path": MODEL_DIR / "glucosidase_model.joblib",
        "chembl_id": "CHEMBL3833502",
        "description": "Human Intestinal Alpha-Glucosidase — diabetes target",
        "activity_threshold": "IC50 ≤ 10 µM",
    },
    "Pancreatic Lipase": {
        "model_path": MODEL_DIR / "lipase_model.joblib",
        "chembl_id": "CHEMBL1812",
        "description": "Human Pancreatic Lipase — diabetes/obesity target",
        "activity_threshold": "IC50 ≤ 10 µM",
    },
    "ACE": {
        "model_path": MODEL_DIR / "ace_model.joblib",
        "chembl_id": "CHEMBL1808",
        "description": "Angiotensin-Converting Enzyme — hypertension target",
        "activity_threshold": "IC50 ≤ 10 µM",
    },
    "EGFR": {
        "model_path": MODEL_DIR / "egfr_model.joblib",
        "chembl_id": "CHEMBL203",
        "description": "Epidermal Growth Factor Receptor — lung cancer target",
        "activity_threshold": "IC50 ≤ 1 µM",
    },
    "HER2": {
        "model_path": MODEL_DIR / "her2_model.joblib",
        "chembl_id": "CHEMBL1824",
        "description": "Human Epidermal Growth Factor Receptor 2 — breast cancer target",
        "activity_threshold": "IC50 ≤ 1 µM",
    },
    "VEGFR2": {
        "model_path": MODEL_DIR / "vegfr2_model.joblib",
        "chembl_id": "CHEMBL279",
        "description": "Vascular Endothelial Growth Factor Receptor 2 — anti-angiogenesis",
        "activity_threshold": "IC50 ≤ 1 µM",
    },
    "BRAF": {
        "model_path": MODEL_DIR / "braf_model.joblib",
        "chembl_id": "CHEMBL5145",
        "description": "BRAF Kinase (V600E) — melanoma target",
        "activity_threshold": "IC50 ≤ 1 µM",
    },
    "CDK2": {
        "model_path": MODEL_DIR / "cdk2_model.joblib",
        "chembl_id": "CHEMBL301",
        "description": "Cyclin-Dependent Kinase 2 — cell cycle / cancer target",
        "activity_threshold": "IC50 ≤ 1 µM",
    },
}

# Target name mapping for modules
TARGET_NAME_MAP = {
    "dpp4": "dpp4", "alphaamylase": "amylase", "alphaglucosidase": "glucosidase",
    "pancreaticlipase": "lipase", "ace": "ace", "egfr": "egfr",
    "her2": "her2", "vegfr2": "vegfr2", "braf": "braf", "cdk2": "cdk2",
}


def target_to_key(target: str) -> str:
    """Convert display target name to internal key."""
    key = target.lower().replace("-", "").replace(" ", "")
    return TARGET_NAME_MAP.get(key, key)


# ═══════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model(target: str):
    """Load a trained ML model from disk."""
    info = TARGETS.get(target)
    if info is None:
        return None
    model_path = info["model_path"]
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data
def load_training_fingerprints(target: str) -> np.ndarray | None:
    """Load precomputed training fingerprints for applicability domain check."""
    prefix = target_to_key(target)
    fp_path = DATA_DIR / f"{prefix}_train_fingerprints.npy"
    if not fp_path.exists():
        return None
    return np.load(fp_path)


# ═══════════════════════════════════════════════════════════════
# UI Helpers
# ═══════════════════════════════════════════════════════════════

def mol_to_image(smiles: str) -> bytes | None:
    """Render a molecule image from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(350, 250))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_pipeline(target: str) -> PredictionPipeline | None:
    """Build a PredictionPipeline for the given target."""
    model = load_model(target)
    if model is None:
        return None
    train_fps = load_training_fingerprints(target)
    return PredictionPipeline(
        model=model,
        target=target,
        target_key=target_to_key(target),
        training_fingerprints=train_fps,
    )


def display_single_result(result: PredictionResult, run_dock: bool) -> None:
    """Display a single prediction result in the Streamlit UI."""
    if result.evaluation_status == EvaluationStatus.INVALID_INPUT:
        st.error(f"Invalid input: {result.validation.error}")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        if result.smiles:
            img = mol_to_image(result.smiles)
            if img:
                st.image(img, caption=result.smiles[:60])

    with col2:
        # Flag display (top priority)
        if result.flag:
            if result.flag == "INVESTIGATE":
                st.warning(f"⚡ {result.flag}: {result.flag_reason}")
            elif result.flag == "CAUTION":
                st.info(f"⚠️ {result.flag}: {result.flag_reason}")
            else:
                st.info(f"🔬 {result.flag}: {result.flag_reason}")

        # Primary score
        if result.final_score is not None:
            st.markdown(f"### Score: {result.bioactivity_percent:.1f}%")
            st.caption(f"Primary method: **{result.primary_method}**")

            if result.final_score >= 0.7:
                st.success("🟢 **Strong candidate** — high predicted activity")
            elif result.final_score >= 0.5:
                st.markdown("🟡 **Moderate candidate** — potential activity")
            elif result.final_score >= 0.3:
                st.markdown("🟠 **Weak candidate** — low predicted activity")
            else:
                st.markdown("🔴 **Likely inactive** against this target")
        else:
            st.error("Could not compute prediction score.")
            return

        # Evaluation status badge
        status = result.evaluation_status
        if status == EvaluationStatus.FULLY_EVALUATED:
            st.caption("✅ Fully evaluated")
        elif status == EvaluationStatus.PARTIALLY_EVALUATED:
            st.caption("⚠️ Partially evaluated — docking skipped")
        elif status == EvaluationStatus.FAILED_PROCESSING:
            st.caption("❌ Processing failed")

        st.markdown("---")

        # ── Score Breakdown ──
        st.markdown("#### Score Breakdown")

        # ML Score
        ml_score = result.ml_score
        if ml_score is not None:
            model_type = result.ml_prediction.details.get("model_type", "")
            if model_type == "peptide_specific":
                ml_label = "Peptide ML Model"
                ad_str = " (✅ trained on peptide IC50 data)"
            else:
                ad = result.ad_score
                in_dom = result.in_domain
                dom_icon = "✅" if in_dom else "⚠️"
                ad_str = f" ({dom_icon} AD: {ad:.3f})" if ad is not None else ""
                ml_label = "ML Model"
            st.markdown(f"**{ml_label}:** {ml_score*100:.1f}%{ad_str}")
            if model_type != "peptide_specific" and not result.in_domain:
                st.caption("Outside applicability domain — ML score less reliable")

        # Physicochemical Score
        physico = result.physico_score
        if physico is not None:
            st.markdown(f"**Physicochemical:** {physico*100:.1f}%")
            with st.expander("Component breakdown"):
                components = result.physicochemical.details.get("components", {})
                if components:
                    for comp, score in components.items():
                        label = comp.replace("_", " ").title()
                        st.progress(score, text=f"{label}: {score:.2f}")
        elif result.ligand_type == LigandType.PEPTIDE:
            st.caption("Physicochemical scoring not available for this target")

        # Docking Feasibility
        feasibility = result.docking_feasibility.details
        if feasibility:
            rec = feasibility.get("recommendation", "unknown")
            metrics = feasibility.get("metrics", {})
            rot = metrics.get("rotatable_bonds", "?")
            st.markdown(f"**Docking Feasibility:** {rec} (rotatable bonds: {rot})")

        # Docking Score
        if result.docking_affinity is not None:
            aff = result.docking_affinity
            norm = result.docking_score
            st.markdown(f"**Docking:** {aff:.1f} kcal/mol (normalized: {norm*100:.1f}%)")
        elif result.docking.status == "skipped":
            st.caption(f"Docking skipped: {result.docking_skip_reason}")
        elif result.docking.status == "failed":
            st.caption(f"Docking failed: {result.docking.error}")
        elif run_dock and result.docking.status == "not_requested":
            pass  # User didn't request

        # Confidence
        if result.prediction_confidence is not None:
            st.markdown(f"**Prediction Confidence:** {result.prediction_confidence:.2f}")


# ═══════════════════════════════════════════════════════════════
# Streamlit App
# ═══════════════════════════════════════════════════════════════

def main():
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="BioActive-MultiTarget",
        page_icon="🧬",
        layout="wide",
    )

    st.title("🧬 BioActive-MultiTarget")
    st.markdown(
        "**Target-specific bioactivity screening** — ML + Physicochemical + Docking"
    )

    # ── Sidebar ──
    st.sidebar.header("Target Selection")
    available_targets = [t for t, info in TARGETS.items() if info["model_path"].exists()]

    if not available_targets:
        st.error("No trained models found. Run the training pipeline first.")
        return

    selected_target = st.sidebar.selectbox("Select Target", available_targets)
    target_info = TARGETS[selected_target]
    st.sidebar.markdown(f"**ChEMBL ID:** {target_info['chembl_id']}")
    st.sidebar.markdown(f"**Activity cutoff:** {target_info['activity_threshold']}")
    st.sidebar.markdown(f"_{target_info['description']}_")

    # Docking status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**System Status:**")
    docking_ready = is_docking_available()
    if docking_ready:
        st.sidebar.success("✅ Docking: Available")
    else:
        st.sidebar.warning("⏳ Docking: Not configured")
        st.sidebar.caption("Install Vina + receptors for structure-based scoring")

    model = load_model(selected_target)
    if model is None:
        st.error(f"Failed to load model for {selected_target}")
        return

    # ── Main Tabs ──
    tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

    # ══════════════════════════════════════════════════════════
    # SINGLE PREDICTION
    # ══════════════════════════════════════════════════════════
    with tab_single:
        st.subheader(f"Predict {selected_target} Inhibition")

        input_mode = st.radio(
            "Input type", ["SMILES", "Peptide Sequence"], horizontal=True
        )

        col_input, col_options = st.columns([3, 1])
        with col_input:
            if input_mode == "SMILES":
                user_input = st.text_input(
                    "Enter SMILES",
                    placeholder="e.g., CC(=O)OC1=CC=CC=C1C(=O)O",
                )
            else:
                user_input = st.text_input(
                    "Enter Peptide Sequence (1-letter codes)",
                    placeholder="e.g., IPP, LKPNM, VHLEIP",
                )
        with col_options:
            run_dock = st.checkbox(
                "Run Docking",
                value=False,
                disabled=not docking_ready,
                help="Run AutoDock Vina. Subject to feasibility assessment.",
            )

        predict_btn = st.button("Predict", type="primary")

        if predict_btn and user_input.strip():
            pipeline = build_pipeline(selected_target)
            if pipeline is None:
                st.error("Failed to build prediction pipeline")
                return

            with st.spinner("Computing predictions..."):
                result = pipeline.predict(user_input.strip(), run_docking=run_dock)

            display_single_result(result, run_dock)

        elif predict_btn:
            st.warning("Please enter a value.")

    # ══════════════════════════════════════════════════════════
    # BATCH PREDICTION
    # ══════════════════════════════════════════════════════════
    with tab_batch:
        st.subheader("Batch Prediction")

        batch_mode = st.radio(
            "Batch mode",
            ["Single Target", "All Targets"],
            horizontal=True,
            help="'All Targets' scores each compound against all available models.",
        )

        input_type = st.radio(
            "Input type",
            ["SMILES CSV", "Peptide Sequence CSV"],
            horizontal=True,
            help="Upload a CSV with a 'smiles' or 'sequence' column.",
        )

        if input_type == "SMILES CSV":
            st.markdown("Upload a CSV with a column named `smiles`.")
        else:
            st.markdown("Upload a CSV with a column named `sequence` (1-letter amino acid codes).")

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                return

            # Find the right column
            if input_type == "Peptide Sequence CSV":
                col_name = next((c for c in input_df.columns if c.lower() == "sequence"), None)
                if col_name is None:
                    st.error("CSV must contain a column named 'sequence'.")
                    return
                st.info(f"Found {len(input_df)} peptide sequences.")
            else:
                col_name = next((c for c in input_df.columns if c.lower() == "smiles"), None)
                if col_name is None:
                    st.error("CSV must contain a column named 'smiles'.")
                    return
                st.info(f"Found {len(input_df)} compounds.")

            # Docking option for batch
            col_dock_opt, col_dock_info = st.columns([1, 2])
            with col_dock_opt:
                batch_dock = st.checkbox(
                    "Include Docking",
                    value=False,
                    disabled=not docking_ready,
                    help="Attempt docking for feasible molecules (subject to feasibility assessment).",
                )
            with col_dock_info:
                if batch_dock:
                    st.caption(
                        "Docking will only run for molecules that pass the feasibility assessment. "
                        "Large/flexible peptides will be marked as PARTIALLY_EVALUATED."
                    )

            if st.button("Run Batch Prediction", type="primary"):
                targets_to_predict = available_targets if batch_mode == "All Targets" else [selected_target]
                inputs = [str(row[col_name]).strip() for _, row in input_df.iterrows()]

                # ── Run unified pipeline for each target ──
                all_results: list[PredictionResult] = []
                total_ops = len(inputs) * len(targets_to_predict)
                progress = st.progress(0)
                op_count = 0

                for t in targets_to_predict:
                    t_pipeline = build_pipeline(t)
                    if t_pipeline is None:
                        # Skip targets without models
                        op_count += len(inputs)
                        progress.progress(min(op_count / total_ops, 1.0))
                        continue

                    def update_progress(done, total):
                        nonlocal op_count
                        progress.progress(min((op_count + done) / total_ops, 1.0))

                    batch_results = run_batch_prediction(
                        t_pipeline, inputs,
                        run_docking=batch_dock,
                        progress_callback=update_progress,
                    )
                    all_results.extend(batch_results)
                    op_count += len(inputs)
                    progress.progress(min(op_count / total_ops, 1.0))

                # Convert to DataFrame with full reporting columns
                results_df = results_to_dataframe(all_results)

                # ── Status Summary ──
                status_counts = results_df["evaluation_status"].value_counts()
                n_full = status_counts.get("fully_evaluated", 0)
                n_partial = status_counts.get("partially_evaluated", 0)
                n_invalid = status_counts.get("invalid_input", 0)
                n_failed = status_counts.get("failed_processing", 0)

                st.success(
                    f"Complete: {n_full} fully evaluated, "
                    f"{n_partial} partially evaluated"
                    + (f", {n_invalid} invalid" if n_invalid else "")
                    + (f", {n_failed} failed" if n_failed else "")
                )

                # ── Display: FULLY EVALUATED (default ranking) ──
                fully_eval = results_df[results_df["evaluation_status"] == "fully_evaluated"]

                if batch_mode == "All Targets" and not fully_eval.empty:
                    # Pivot table view
                    pivot = fully_eval.pivot_table(
                        index="input", columns="target", values="final_score", aggfunc="first"
                    )
                    st.markdown("#### Fully Evaluated — Primary Scores")
                    st.dataframe(
                        pivot.style.format("{:.3f}", na_rep="—").background_gradient(
                            cmap="RdYlGn", vmin=0, vmax=1
                        ), use_container_width=True
                    )
                elif not fully_eval.empty:
                    # Single target ranked list
                    fully_eval = fully_eval.sort_values("bioactivity_%", ascending=False)
                    st.markdown("#### Fully Evaluated Predictions")
                    display_cols = [
                        "input", "bioactivity_%", "final_score", "ml_score",
                        "physico_score", "docking_affinity", "docking_normalized",
                        "prediction_confidence", "primary_method", "flag",
                    ]
                    display_cols = [c for c in display_cols if c in fully_eval.columns]
                    st.dataframe(
                        fully_eval[display_cols].style.format(
                            {c: "{:.3f}" for c in display_cols if c in
                             ("final_score", "ml_score", "physico_score",
                              "docking_normalized", "prediction_confidence")},
                            na_rep="—",
                        ).background_gradient(
                            subset=["bioactivity_%"], cmap="RdYlGn", vmin=0, vmax=100
                        ),
                        use_container_width=True,
                    )

                # ── Display: PARTIALLY EVALUATED ──
                partial_eval = results_df[results_df["evaluation_status"] == "partially_evaluated"]
                if not partial_eval.empty:
                    st.markdown("#### ⚠️ Partially Evaluated (Docking Skipped)")
                    st.caption(
                        "These predictions are based on ML and/or physicochemical analysis only. "
                        "Docking was skipped due to structural limitations."
                    )
                    partial_cols = [
                        "input", "target", "bioactivity_%", "ml_score",
                        "physico_score", "docking_skip_reason",
                        "prediction_confidence", "primary_method",
                    ]
                    partial_cols = [c for c in partial_cols if c in partial_eval.columns]
                    st.dataframe(partial_eval[partial_cols], use_container_width=True)

                # ── Display: INVALID / FAILED ──
                invalid = results_df[results_df["evaluation_status"].isin(
                    ["invalid_input", "failed_processing"]
                )]
                if not invalid.empty:
                    with st.expander(f"❌ Invalid/Failed ({len(invalid)})"):
                        st.dataframe(
                            invalid[["input", "target", "evaluation_status"]],
                            use_container_width=True,
                        )

                # ── Flagged compounds ──
                flagged = results_df[results_df["flag"].notna()]
                if not flagged.empty:
                    st.markdown("#### ⚡ Flagged for Investigation")
                    st.caption("Disagreement between methods — these may be novel candidates")
                    flag_cols = ["input", "target", "ml_score", "physico_score", "flag", "flag_reason"]
                    flag_cols = [c for c in flag_cols if c in flagged.columns]
                    st.dataframe(flagged[flag_cols], use_container_width=True)

                # ── Download full report ──
                csv_output = results_df.to_csv(index=False)
                filename = (
                    "all_targets_predictions.csv"
                    if batch_mode == "All Targets"
                    else f"{selected_target.lower()}_predictions.csv"
                )
                st.download_button(
                    label="Download Full Report CSV",
                    data=csv_output,
                    file_name=filename,
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
