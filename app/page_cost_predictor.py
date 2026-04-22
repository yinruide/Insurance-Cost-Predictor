"""Page 2: Live cost predictor with feature importance."""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── Helper: load best available model ─────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the best available trained model."""
    model_paths = [
        ("XGBoost",       "saved_models/xgboost_model.pkl"),
        ("Random Forest", "saved_models/random_forest.pkl"),
    ]
    for name, path in model_paths:
        if os.path.exists(path):
            return name, joblib.load(path)
    return None, None

def preprocess_input(age, sex, bmi, children, smoker, region):
    """Encode user inputs to match training encoding."""
    sex_enc    = 0 if sex == "Female" else 1
    smoker_enc = 1 if smoker == "Yes" else 0
    region_map = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
    region_enc = region_map[region]
    return pd.DataFrame([[age, sex_enc, bmi, children, smoker_enc, region_enc]],
                        columns=["age", "sex", "bmi", "children", "smoker", "region"])

def plot_feature_importance(model, model_name):
    """Bar chart of feature importances."""
    features = ["age", "sex", "bmi", "children", "smoker", "region"]
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e63946" if features[i] == "smoker" else "#457b9d" for i in sorted_idx]
    ax.barh([features[i] for i in sorted_idx], importances[sorted_idx], color=colors)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance — {model_name}")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig

# ── Page ───────────────────────────────────────────────────────────────────────
def show():
    st.title("Live Insurance Cost Predictor")
    st.markdown("Enter your details below to get a personalized insurance cost estimate.")

    model_name, model = load_model()

    if model is None:
        st.error(
            "No trained model found. Please run `models/random_forest.py` or "
            "`models/xgboost_model.py` first to generate a saved model."
        )
        return

    st.info(f"Using model: **{model_name}**")
    st.markdown("---")

    # ── Input form ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        age      = st.slider("Age", min_value=18, max_value=100, value=30)
        sex      = st.selectbox("Sex", ["Female", "Male"])
        bmi      = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                                   help="Body Mass Index. Healthy range: 18.5–24.9")
    with col2:
        children = st.slider("Number of Children", min_value=0, max_value=10, value=0)
        smoker   = st.selectbox("Smoker?", ["No", "Yes"])
        region   = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    st.markdown("---")

    if st.button("Predict My Insurance Cost", use_container_width=True):
        input_df  = preprocess_input(age, sex, bmi, children, smoker, region)
        prediction = model.predict(input_df)[0]

        # Result
        st.success(f"### Estimated Annual Insurance Cost: **${prediction:,.2f}**")

        # Smoker warning
        if smoker == "Yes":
            st.warning(
                "Smoking is the single largest driver of insurance costs. "
                "Quitting could save you thousands of dollars per year."
            )

        st.markdown("---")

        # ── Feature importance ────────────────────────────────────────────────
        st.subheader("What Drives Your Cost?")
        st.markdown(
            "The chart below shows how much each factor influences the model's prediction. "
            "Red = highest impact feature."
        )
        fig = plot_feature_importance(model, model_name)
        st.pyplot(fig)

        # Breakdown table
        st.subheader("Your Profile Summary")
        summary = pd.DataFrame({
            "Feature": ["Age", "Sex", "BMI", "Children", "Smoker", "Region"],
            "Your Value": [age, sex, f"{bmi:.1f}", children, smoker, region],
        })
        st.table(summary)