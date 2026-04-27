"""
app.py
Multi-page application shell.

Run with:
    streamlit run app/app.py
"""

import os

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
