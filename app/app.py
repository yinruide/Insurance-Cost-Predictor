"""
app.py
Multi-page application shell.

Run with:
    streamlit run app/app.py
"""

import os
import sys
from pathlib import Path

# Prevent OpenMP / BLAS runtimes from over-initializing threads under Streamlit
# on macOS, which can crash startup with pthread/OpenMP errors.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import streamlit as st

from page_cost_predictor import render_page as render_predictor
from page_data_exploration import render_page as render_exploration
from page_model_comparison import render_page as render_comparison
from shared import PALETTE, inject_global_styles

# Make models/ importable for the artifact recovery path
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_MODELS_DIR = _ROOT / "models"
sys.path.insert(0, str(_MODELS_DIR))

_SAVED_DIR = _ROOT / "saved_models"


@st.cache_resource(show_spinner=False)
def ensure_artifacts() -> dict:
    """
    Verify all model artifacts the app depends on exist on disk.
    If any are missing, retrain them on first launch.

    Cached for the lifetime of the Streamlit session, so this runs at most
    once per app boot regardless of which page the user opens first.

    Returns a dict summarizing what was retrained, for optional display.
    """
    retrained = []

    # ── Block 1: regression models (each writes its own metrics + pkl/pt) ──
    block1_checks = [
        ("Linear Regression", _SAVED_DIR / "linear_metrics.json",
         "linear_regression", "fit_linear_regression"),
        ("Random Forest",     _SAVED_DIR / "random_forest.pkl",
         "random_forest", "fit_random_forest"),
        ("XGBoost",           _SAVED_DIR / "xgboost_model.pkl",
         "xgboost_model", "fit_xgboost"),
        ("MLP",               _SAVED_DIR / "mlp_regressor.pt",
         "mlp", "fit_mlp"),
        ("Quantile Reg.",     _SAVED_DIR / "quantile_metrics.json",
         "quantile_regression", "fit_quantile_models"),
        ("MDN",               _SAVED_DIR / "mdn_regressor.pt",
         "mdn", "fit_mdn"),
    ]

    # ── Block 2: classifier + subgroup regressors ──
    block2_files = [
        _SAVED_DIR / "lr_smoker_classifier.pkl",
        _SAVED_DIR / "rf_smoker_classifier.pkl",  # either lr or rf is fine
        _SAVED_DIR / "rf_regressor_smoker.pkl",
        _SAVED_DIR / "rf_regressor_nonsmoker.pkl",
    ]
    # We need: at least one classifier exists AND both subgroup regressors exist
    has_any_classifier = block2_files[0].exists() or block2_files[1].exists()
    has_both_regressors = block2_files[2].exists() and block2_files[3].exists()
    block2_ok = has_any_classifier and has_both_regressors

    # Quick path: nothing to do
    missing_block1 = [b for b in block1_checks if not b[1].exists()]
    if not missing_block1 and block2_ok:
        return {"retrained": [], "skipped": True}

    # Slow path: train what's missing, with visible progress
    with st.status(
        "First-time setup: training missing model artifacts…",
        expanded=True,
    ) as status:
        for label, path, module_name, fn_name in missing_block1:
            if path.exists():
                continue
            st.write(f"Training {label}…")
            module = __import__(module_name)
            getattr(module, fn_name)()
            retrained.append(label)

        if not block2_ok:
            st.write("Training Block 2 (classifier + subgroup regressors)…")
            from block2_classifier import run_block2
            run_block2()
            retrained.append("Block 2 pipeline")

        status.update(
            label=f"Setup complete — retrained: {', '.join(retrained)}",
            state="complete",
            expanded=False,
        )

    return {"retrained": retrained, "skipped": False}


_PAGES = {
    "Cost Predictor":   render_predictor,
    "Data Exploration": render_exploration,
    "Model Comparison": render_comparison,
}

_ICONS = {
    "Cost Predictor":   "💡",
    "Data Exploration": "📊",
    "Model Comparison": "🏆",
}


def main():
    st.set_page_config(
        page_title="Insurance Cost Predictor",
        page_icon="💡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_styles()

    # Recover any missing model artifacts on first launch.
    # Cached, so this is a no-op after the first page load.
    ensure_artifacts()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1.25rem;">
                <div style="font-size:0.65rem; font-weight:700; letter-spacing:0.14em;
                            text-transform:uppercase; color:#c96c1a; margin-bottom:0.3rem;">
                    Final Project
                </div>
                <div style="font-family:'Fraunces',Georgia,serif; font-size:1.05rem;
                            font-weight:700; color:#f0ece2; line-height:1.3;">
                    Medical Insurance<br>Cost Predictor
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selection = st.radio(
            "Navigate",
            options=list(_PAGES.keys()),
            format_func=lambda k: f"{_ICONS[k]}  {k}",
            label_visibility="collapsed",
        )

        st.markdown(
            """
            <div style="margin-top: 2rem; border-top: 1px solid rgba(255,255,255,0.1);
                        padding-top: 1rem;">
                <div style="font-size:0.72rem; color:#6e6a60; line-height:1.6;">
                    <strong style="color:#9aa0ab;">Demo flow</strong><br>
                    Start on Cost Predictor,<br>
                    show Data Exploration,<br>
                    close with Model Comparison.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Page render ───────────────────────────────────────────────────────────
    _PAGES[selection]()


if __name__ == "__main__":
    main()