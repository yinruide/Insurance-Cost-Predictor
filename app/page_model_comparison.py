"""
page_model_comparison.py
Model Comparison — Page 3

Shows held-out metrics for all five models, highlights the best performer,
explains the MDN as the from-scratch implementation, and includes the
quantile interval diagnostics chart.
"""

import pandas as pd
import streamlit as st

from shared import (
    PALETTE,
    SAVED_DIR,
    card,
    load_comparison_metrics,
    metric_tile,
    page_header,
    plot_model_comparison,
    plot_prediction_interval,
    tags,
)


def _leaderboard_html(df: pd.DataFrame) -> str:
    """Custom HTML leaderboard — best row highlighted, no raw st.dataframe."""
    best_model = df.iloc[0]["Model"]
    rows = ""
    for _, row in df.iterrows():
        is_best = row["Model"] == best_model
        row_bg = PALETTE["accent_bg"] if is_best else PALETTE["surface"]
        badge = (
            f'<span style="display:inline-block; padding:0.15rem 0.45rem; '
            f'background:{PALETTE["accent"]}; color:white; border-radius:4px; '
            f'font-size:0.65rem; font-weight:700; letter-spacing:0.05em; '
            f'text-transform:uppercase; margin-left:0.4rem; vertical-align:middle;">Best</span>'
            if is_best else ""
        )
        r2_color = PALETTE["green"] if is_best else PALETTE["ink"]
        r2_weight = "700" if is_best else "600"
        rows += (
            f'<tr style="background:{row_bg};">'
            f'<td style="padding:0.7rem 0.875rem; border-bottom:1px solid {PALETTE["border"]}; color:{PALETTE["ink"]}; vertical-align:middle;">{row["Model"]}{badge}</td>'
            f'<td style="padding:0.7rem 0.875rem; border-bottom:1px solid {PALETTE["border"]}; text-align:right; color:{r2_color}; font-weight:{r2_weight}; font-variant-numeric:tabular-nums; vertical-align:middle;">{row["R2"]:.3f}</td>'
            f'<td style="padding:0.7rem 0.875rem; border-bottom:1px solid {PALETTE["border"]}; text-align:right; color:{PALETTE["ink"]}; font-variant-numeric:tabular-nums; vertical-align:middle;">${row["RMSE"]:,.0f}</td>'
            f'<td style="padding:0.7rem 0.875rem; border-bottom:1px solid {PALETTE["border"]}; text-align:right; color:{PALETTE["ink"]}; font-variant-numeric:tabular-nums; vertical-align:middle;">${row["MAE"]:,.0f}</td>'
            f'<td style="padding:0.7rem 0.875rem; border-bottom:1px solid {PALETTE["border"]}; color:{PALETTE["ink_2"]}; font-size:0.78rem; vertical-align:middle;">{row["Notes"]}</td>'
            f'</tr>'
        )
    html = (
        f'<div style="background:{PALETTE["surface"]}; border:1px solid {PALETTE["border"]}; border-radius:12px; overflow:hidden; margin-bottom:1.5rem;">'
        f'<table style="width:100%; border-collapse:collapse; font-size:0.875rem;">'
        f'<thead><tr>'
        f'<th style="background:{PALETTE["surface_2"]}; padding:0.6rem 0.875rem; text-align:left; font-size:0.68rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:{PALETTE["ink_3"]}; border-bottom:1px solid {PALETTE["border"]};">Model</th>'
        f'<th style="background:{PALETTE["surface_2"]}; padding:0.6rem 0.875rem; text-align:right; font-size:0.68rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:{PALETTE["ink_3"]}; border-bottom:1px solid {PALETTE["border"]};">R²</th>'
        f'<th style="background:{PALETTE["surface_2"]}; padding:0.6rem 0.875rem; text-align:right; font-size:0.68rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:{PALETTE["ink_3"]}; border-bottom:1px solid {PALETTE["border"]};">RMSE</th>'
        f'<th style="background:{PALETTE["surface_2"]}; padding:0.6rem 0.875rem; text-align:right; font-size:0.68rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:{PALETTE["ink_3"]}; border-bottom:1px solid {PALETTE["border"]};">MAE</th>'
        f'<th style="background:{PALETTE["surface_2"]}; padding:0.6rem 0.875rem; text-align:left; font-size:0.68rem; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; color:{PALETTE["ink_3"]}; border-bottom:1px solid {PALETTE["border"]};">Notes</th>'
        f'</tr></thead>'
        f'<tbody>{rows}</tbody>'
        f'</table></div>'
    )
    return html


def render_page():
    page_header(
        "Model Comparison",
        "Results on Held-Out Test Data",
        "All metrics are evaluated on the same 20% test split (random_state=12138). "
        "The best overall model by R² is highlighted.",
    )

    with st.spinner("Loading model metrics…"):
        bundle = load_comparison_metrics()

    leaderboard = bundle["leaderboard"].copy()
    best        = leaderboard.iloc[0]
    qr          = bundle["quantile"]
    mdn         = bundle["mdn"]

    # ── Top-line metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        metric_tile("Best R²",         f"{best['R2']:.3f}",                  best["Model"])
    with c2:
        metric_tile("Best RMSE",       f"${leaderboard['RMSE'].min():,.0f}",
                    leaderboard.loc[leaderboard["RMSE"].idxmin(), "Model"])
    with c3:
        metric_tile("80% coverage",    f"{qr['interval_80_coverage']:.1%}",  "Quantile regression")
    with c4:
        metric_tile("MDN R²",          f"{mdn['r2']:.3f}",                   "From-scratch model")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Leaderboard + chart ───────────────────────────────────────────────────
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<span class="section-label">Full leaderboard</span>', unsafe_allow_html=True)
        st.markdown(_leaderboard_html(leaderboard), unsafe_allow_html=True)

    with right:
        st.markdown('<span class="section-label">R² comparison</span>', unsafe_allow_html=True)
        st.pyplot(plot_model_comparison(leaderboard))

    # ── Takeaway card ─────────────────────────────────────────────────────────
    card(
        "Why we use multiple models",
        f"""
        <strong>Linear regression</strong> gives an interpretable baseline on log-transformed charges.<br>
        <strong>Random Forest</strong> captures nonlinear feature interactions and is the top performer
        (R² = {best['R2']:.3f}) — this is the model used in the live predictor.<br>
        <strong>MLP</strong> provides a neural alternative with comparable performance.<br>
        <strong>Quantile regression</strong> adds calibrated uncertainty bounds (80% interval
        coverage: {qr['interval_80_coverage']:.1%}).<br>
        <strong>MDN</strong> is our from-scratch implementation: a PyTorch mixture density network
        that models charge distributions as a mixture of Gaussians — suited to the bimodal
        smoker/non-smoker structure visible in the EDA.
        """,
    )

    st.divider()

    # ── Quantile interval diagnostics ─────────────────────────────────────────
    st.markdown('<span class="section-label">Uncertainty diagnostics · quantile regression intervals</span>', unsafe_allow_html=True)
    try:
        quantile_predictions = pd.read_csv(SAVED_DIR / "quantile_predictions.csv")
        st.pyplot(plot_prediction_interval(quantile_predictions))
        st.caption(
            "First 50 test examples. Gold band = 80% predicted interval; "
            "line = median prediction; dots = actual charges."
        )
    except FileNotFoundError:
        st.info("Run quantile_regression.py to generate interval diagnostics.")

    st.divider()

    # ── MDN deep-dive ─────────────────────────────────────────────────────────
    st.markdown('<span class="section-label">From-scratch implementation · Mixture Density Network</span>', unsafe_allow_html=True)
    tags("PyTorch", f"{mdn['n_components']} Gaussian components", "Trained from scratch", "NLL objective")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3, gap="small")
    with d1:
        metric_tile("MDN R²",      f"{mdn['r2']:.3f}",          "Held-out test set")
    with d2:
        metric_tile("MDN RMSE",    f"${mdn['rmse']:,.0f}",       f"{mdn['n_components']} components")
    with d3:
        metric_tile("Best val NLL", f"{mdn['best_val_nll']:.2f}", "Training objective (NLL)")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    card(
        None,
        """
        The MDN outputs a mixture of Gaussians over the charge distribution rather than
        a single point estimate. This is well-suited to insurance data because the
        smoker/non-smoker split creates a <strong>bimodal outcome distribution</strong>
        — a single Gaussian cannot represent it well. We implemented the full forward
        pass, NLL loss, and training loop from scratch in PyTorch with no high-level
        mixture API. Predictions are recovered as the expectation of the mixture.
        """,
    )


def main():
    from shared import inject_global_styles
    st.set_page_config(
        page_title="Model Comparison — Insurance",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_global_styles()
    render_page()


if __name__ == "__main__":
    main()
