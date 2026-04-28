"""
page_cost_predictor.py
Live Cost Predictor — Page 2

UX decisions
────────────
• st.form wrapper: sliders don't trigger reruns until the CTA is clicked.
• st.session_state: the last prediction persists across reruns so tweaking
  one input after submitting doesn't blank the result panel.
• Result panel is always visible on the right — either an empty-state
  prompt or the actual output. The user always knows where to look.
• No decorative chips that look like buttons.
"""

import streamlit as st

from shared import (
    block2_summary_text,
    card,
    empty_state,
    make_prediction,
    metric_tile,
    page_header,
    plot_feature_impacts,
    result_panel,
    routing_card,
)

_REGIONS = ["northeast", "northwest", "southeast", "southwest"]


def _profile_form() -> dict:
    """Compact two-column profile form inside an st.form."""
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        age      = st.slider("Age", 18, 64, 35)
        bmi      = st.slider("BMI", 15.0, 54.0, 27.5, step=0.1)
        children = st.select_slider("Children", options=list(range(6)), value=1)
    with col2:
        sex    = st.selectbox("Sex", ["female", "male"])
        region = st.selectbox("Region", _REGIONS)
        smoker_status = st.radio(
            "Smoking status",
            ["no", "yes", "unknown"],
            horizontal=True,
            help=(
                "Select 'unknown' to let the model estimate smoker probability "
                "from the rest of the profile using the Block 2 classifier."
            ),
        )
    return {
        "age": age, "bmi": bmi, "children": children,
        "sex": sex, "region": region, "smoker_status": smoker_status,
    }


def _render_result(pred: dict):
    """Render the full result panel for a computed prediction."""
    # Primary output — big number + range bar
    result_panel(pred["estimate"], pred["q10"], pred["q90"], pred["q50"])

    # Two supporting metrics
    c1, c2 = st.columns(2, gap="small")
    with c1:
        if pred["segment"] == "weighted blend":
            metric_tile(
                "Smoker probability",
                f"{pred['smoker_probability']:.0%}",
                "estimated from demographics",
            )
        else:
            label = "Smoker path" if pred["segment"] == "smoker segment" else "Non-smoker path"
            metric_tile(label, "used directly", "status was provided")
    with c2:
        metric_tile(
            "Scenario spread",
            f"${abs(pred['smoker_cost'] - pred['nonsmoker_cost']):,.0f}",
            f"smoker ${pred['smoker_cost']:,.0f} · non-smoker ${pred['nonsmoker_cost']:,.0f}",
        )

    # Routing explanation
    routing_card(pred["segment"], block2_summary_text(pred))

    # Feature impacts
    st.markdown(
        '<span class="section-label">What influenced this estimate</span>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Heuristic local drivers — feature importance × deviation from dataset median. "
        "Not a formal SHAP value; use as directional guidance."
    )
    st.pyplot(plot_feature_impacts(pred["impacts"]))


def render_page():
    page_header(
        "Live Predictor",
        "Annual Cost Estimate",
        "Enter a profile and get a predicted annual insurance cost, "
        "an 80% uncertainty band, and an explanation of which model path was used.",
    )

    left, right = st.columns([1, 1.15], gap="large")

    # ── Left: form ────────────────────────────────────────────────────────────
    with left:
        st.markdown('<span class="section-label">Profile inputs</span>', unsafe_allow_html=True)
        with st.form("profile_form"):
            profile = _profile_form()
            submitted = st.form_submit_button(
                "Estimate Annual Cost",
                use_container_width=True,
            )

        if submitted:
            with st.spinner("Running predictor…"):
                st.session_state["last_prediction"] = make_prediction(profile)

        # How-it-works card — always visible, below the form
        card(
            "Two-stage pipeline (Block 2)",
            """
            If smoker status is <strong>known</strong>, the profile routes directly to the
            matching subgroup Random Forest regressor.<br><br>
            If <strong>unknown</strong>, a logistic classifier first estimates smoker
            probability from demographics, then the two subgroup estimates are blended
            proportionally. Uncertainty bounds come from a separately trained
            quantile regression model (q10 / q90).
            """,
        )

    # ── Right: result ─────────────────────────────────────────────────────────
    with right:
        pred = st.session_state.get("last_prediction")
        if pred is None:
            st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
            empty_state(
                "📋",
                "No estimate yet",
                "Fill in the profile on the left and click <strong>Estimate Annual Cost</strong>.",
            )
            st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
            card(
                "Demo tips",
                """
                Try these three profiles to show the full routing logic:<br><br>
                <strong>Low risk</strong> — age 28, BMI 23, no children, non-smoker<br>
                <strong>High risk</strong> — age 55, BMI 36, smoker<br>
                <strong>Unknown</strong> — any profile with smoking status set to
                <em>unknown</em> to demonstrate the classifier step
                """,
            )
        else:
            _render_result(pred)


def main():
    from shared import inject_global_styles
    st.set_page_config(
        page_title="Live Cost Predictor — Insurance",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_global_styles()
    render_page()


if __name__ == "__main__":
    main()