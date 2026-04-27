"""
page_data_exploration.py
Medical Insurance Cost Predictor — Streamlit Page 1: Data Exploration

Integrates all visualizations from eda_utils.py and kmeans.py into an
interactive Streamlit page with tabs, filters, and a dynamic K slider.

Author: Ruide Yin
"""

from pathlib import Path
import sys

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from shared import hero, inject_global_styles, tags

# constants
_HERE = Path(__file__).resolve().parent
DATA_PATH = str(_HERE.parent / "data" / "insurance.csv")

# make exploration/ importable
sys.path.insert(0, str(_HERE.parent / "exploration"))

# project imports
import eda_utils
from kmeans import run_kmeans, K_RANGE


# cached loaders 

@st.cache_data
def load_raw_data(path=DATA_PATH):
    return pd.read_csv(path)


@st.cache_resource
def run_kmeans_cached(path=DATA_PATH, best_k=2):
    return run_kmeans(path=path, best_k=best_k)


# helper 

def show_fig(fig):
    """Display a matplotlib figure in Streamlit and close it to free memory."""
    st.pyplot(fig)
    plt.close(fig)


# sidebar filters 

def sidebar_filters(df):
    """
    Sidebar multi-select filters for region and smoker status.
    Returns a filtered copy of df.
    """
    st.sidebar.header("Filters")

    regions = st.sidebar.multiselect(
        "Region",
        options=sorted(df["region"].unique()),
        default=sorted(df["region"].unique()),
    )
    smoker_opts = st.sidebar.multiselect(
        "Smoker",
        options=["yes", "no"],
        default=["yes", "no"],
    )

    filtered = df[df["region"].isin(regions) & df["smoker"].isin(smoker_opts)]

    if len(filtered) == 0:
        st.sidebar.warning("No data matches current filters.")
    else:
        st.sidebar.caption(f"Showing **{len(filtered)}** / {len(df)} records")

    return filtered


# page sections 

def section_dataset_overview(df):
    """Section 0: raw data preview and summary statistics."""
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{len(df):,}")
    col2.metric("Features", f"{df.shape[1] - 1}")  # exclude target
    col3.metric("Smokers", f"{(df['smoker'] == 'yes').sum()}")
    col4.metric("Avg Charge", f"${df['charges'].mean():,.0f}")

    with st.expander("Preview raw data", expanded=False):
        st.dataframe(df.head(20), width="stretch")

    with st.expander("Summary statistics", expanded=False):
        st.dataframe(df.describe().round(2), width="stretch")


def section_eda(df):
    """Section 1: EDA visualizations organized into tabs."""
    st.header("Exploratory Data Analysis")

    tab_dist, tab_rel, tab_corr = st.tabs([
        "Feature Distributions",
        "Feature vs Charges",
        "Correlations",
    ])

    # Tab 1: Distributions 
    with tab_dist:
        st.subheader("Numerical Feature Distributions")
        st.markdown(
            "Age is roughly uniform; BMI is approximately normal; "
            "charges are **strongly right-skewed** (skew ≈ 1.5), "
            "suggesting a log-transform may help regression models."
        )
        show_fig(eda_utils.plot_numerical_distributions(df))

        st.subheader("Charges vs Log-Charges")
        st.markdown(
            "Log-transforming charges reduces skewness significantly, "
            "making the distribution closer to normal."
        )
        show_fig(eda_utils.plot_charges_log_comparison(df))

        st.subheader("Categorical Feature Counts")
        st.markdown(
            "Sex is roughly balanced; smokers are a **minority (~20%)**; "
            "region distribution is approximately even."
        )
        show_fig(eda_utils.plot_categorical_counts(df))

    # Tab 2: Feature vs Charges 
    with tab_rel:
        st.subheader("Categorical Features vs Charges")
        st.markdown(
            "Smoker status dominates: smokers have dramatically higher "
            "charges. Sex and region show modest differences."
        )
        show_fig(eda_utils.plot_categorical_vs_charges(df))

        st.subheader("Numerical Features vs Charges (colored by smoker)")
        st.markdown(
            "Two distinct bands are visible in the scatter plots — "
            "smokers form a separate, higher-cost cluster."
        )
        show_fig(eda_utils.plot_scatter_vs_charges(df))

        st.subheader("Charges by Number of Children")
        show_fig(eda_utils.plot_charges_by_children(df))

        st.subheader("Age vs Charges — Linear Fit by Smoker Status")
        st.markdown(
            "Both groups show a positive age–charges trend, but the "
            "smoker regression line sits much higher with a steeper slope."
        )
        show_fig(eda_utils.plot_age_vs_charges_regression(df))

        st.subheader("Charges by Age Group × Smoker Status")
        show_fig(eda_utils.plot_charges_by_age_group(df))

    # Tab 3: Correlations 
    with tab_corr:
        st.subheader("Pearson Correlation Matrix")
        st.markdown(
            "Smoker status has the highest correlation with charges "
            "(r ≈ 0.79). Age and BMI contribute moderately."
        )
        show_fig(eda_utils.plot_correlation_heatmap(df))

        st.subheader("Charge Distribution by Smoker Status (KDE)")
        st.markdown(
            "The charge distributions are **near-disjoint**: smokers "
            "center around $30k+ while non-smokers peak below $10k. "
            "This bimodality motivates our Block 2 stratified pipeline."
        )
        show_fig(eda_utils.plot_smoker_charge_kde(df))

        st.subheader("Pairplot (Age, BMI, Charges)")
        show_fig(eda_utils.plot_pairplot(df))


def section_kmeans(df_full):
    """
    Section 2: K-Means clustering analysis.
    Uses the FULL (unfiltered) dataset — clustering on a filtered subset
    would distort the results.
    """
    st.header("K-Means Clustering Analysis")
    st.markdown(
        "We run K-Means on **age, BMI, and charges** (intentionally "
        "excluding smoker status) to test whether the algorithm can "
        "recover the smoker / non-smoker split without supervision."
    )

    # K slider 
    best_k = st.slider(
        "Select number of clusters (K)",
        min_value=int(min(K_RANGE)),
        max_value=int(max(K_RANGE)),
        value=2,
        help="K=2 is optimal per Elbow + Silhouette analysis.",
    )

    results = run_kmeans_cached(path=DATA_PATH, best_k=best_k)

    st.caption(
        "Use the slider, then read left to right: first see how the clusters change, "
        "then validate them, then review why K=2 is the recommended default."
    )

    # Put the dynamic output first so the slider has an obvious effect.
    tab_result, tab_validate, tab_select = st.tabs([
        "Cluster Results",
        "Cluster Validation",
        "Model Selection",
    ])

    with tab_result:
        st.subheader("PCA Projection of Clusters")
        st.markdown(
            f"Cluster assignments with **K = {best_k}** projected onto "
            "the first two principal components."
        )
        show_fig(results["fig_pca"])
        st.caption(
            "Each color corresponds to one cluster label. The black X markers show the learned centroids."
        )

        st.subheader("Silhouette Samples Plot")
        st.markdown(
            f"Per-sample silhouette coefficients grouped by cluster. "
            f"Overall silhouette score: **{results['best_sil']:.3f}**."
        )
        show_fig(results["fig_sil_samples"])

    with tab_validate:
        st.subheader("Cluster vs Actual Smoker Status")
        st.markdown(
            "The cross-tabulation shows near-perfect alignment between "
            "K-Means clusters and true smoker labels — unsupervised "
            "evidence supporting our Block 2 stratified pipeline."
        )
        show_fig(results["fig_vs_smoker"])

        st.subheader("Feature Distributions by Cluster")
        st.markdown(
            "Charges clearly separate the two clusters, with the "
            "high-charge cluster corresponding to smokers."
        )
        show_fig(results["fig_box"])

    with tab_select:
        st.subheader("Elbow Method")
        st.markdown(
            "Inertia (within-cluster SSE) drops sharply at K=2, "
            "with diminishing returns beyond that point."
        )
        show_fig(results["fig_elbow"])

        st.subheader("Silhouette Scores by K")
        st.markdown(
            "K=2 achieves the highest silhouette score, confirming "
            "two natural clusters in the data."
        )
        show_fig(results["fig_silhouette"])


# main 
def render_page():
    """Render the exploration page inside the full app shell."""
    from shared import page_header
    page_header(
        "Data Exploration",
        "Dataset Patterns & EDA",
        "Interactive exploration of the Kaggle Medical Insurance dataset (1,338 records). "
        "Use the sidebar filters to slice by region and smoker status.",
    )
    tags("1,338 records", "7 features", "EDA + K-Means clustering")
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Load full dataset
    df_full = load_raw_data()

    # Sidebar filters → filtered copy for EDA
    df_filtered = sidebar_filters(df_full)

    # Render sections
    section_dataset_overview(df_filtered)

    st.divider()
    section_eda(df_filtered)

    st.divider()
    section_kmeans(df_full)  # always use full data for clustering


def main():
    st.set_page_config(
        page_title="Data Exploration — Insurance Cost Predictor",
        page_icon="",
        layout="wide",
    )
    inject_global_styles()
    render_page()


if __name__ == "__main__":
    main()
