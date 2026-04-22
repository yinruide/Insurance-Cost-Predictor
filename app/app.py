"""Streamlit app entry point."""
import streamlit as st

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    layout="wide",
)

# Sidebar navigation
st.sidebar.title("Insurance Cost Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Data Exploration", "Cost Predictor", "Model Comparison"],
)

if page == "Data Exploration":
    from page_data_exploration import show
    show()
elif page == "Cost Predictor":
    from page_cost_predictor import show
    show()
elif page == "Model Comparison":
    from page_model_comparison import show
    show()

