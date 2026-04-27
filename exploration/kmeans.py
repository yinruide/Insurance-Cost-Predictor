"""
kmeans.py
Medical Insurance Cost Predictor — K-Means Clustering Module

Purpose: validate the EDA finding that smoker / non-smoker populations form
         naturally separable clusters in feature space, providing unsupervised
         evidence for the Block 2 stratified pipeline.

Pipeline:
  1. Load raw data → encode categoricals → select numerical features
  2. StandardScaler (K-means is distance-based, needs normalized features)
  3. Elbow + Silhouette analysis to justify K
  4. Fit final K-means model
  5. Visualization functions (return matplotlib Figure objects for Streamlit)

Author: Ruide Yin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

sns.set_style("whitegrid")

RANDOM_STATE = 12138
CLUSTER_FEATURES = ["age", "bmi", "charges"]   # intentionally exclude smoker
K_RANGE = range(2, 9)                           # candidates for elbow / silhouette


# Data Preparation 


def _prepare_clustering_data(path="../data/insurance.csv"):
    """
    Load raw CSV → extract numerical clustering features → scale.

    Returns
    -------
    X_scaled : np.ndarray   — standardized feature matrix (n, 3)
    df_raw   : pd.DataFrame — original unencoded DataFrame (for label comparison)
    scaler   : StandardScaler
    """
    df_raw = pd.read_csv(path)
    X = df_raw[CLUSTER_FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, df_raw, scaler


# Model Selection Helpers 


def _elbow_silhouette(X_scaled):
    """
    Run K-means for each K in K_RANGE. Record inertia and silhouette score.

    Returns
    -------
    inertias    : list[float]
    sil_scores  : list[float]
    """
    inertias = []
    sil_scores = []

    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    return inertias, sil_scores


#  Visualization 


def plot_elbow(inertias):
    """Elbow curve: inertia vs K."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(K_RANGE), inertias, "o-", linewidth=2)
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("Elbow Method")
    ax.set_xticks(list(K_RANGE))
    fig.tight_layout()
    return fig


def plot_silhouette_scores(sil_scores):
    """Bar chart of silhouette scores for each K."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = list(K_RANGE)
    best_idx = sil_scores.index(max(sil_scores))
    colors = ["#2ecc71" if i == best_idx else "#3498db" for i, s in enumerate(sil_scores)]
    ax.bar(ks, sil_scores, color=colors, edgecolor="white")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Analysis")
    ax.set_xticks(ks)
    fig.tight_layout()
    return fig


def plot_clusters_pca(X_scaled, labels, centroids_scaled):
    """
    2-D PCA projection of clusters with centroids marked.

    Parameters
    ----------
    X_scaled         : (n, 3) standardized features
    labels           : (n,)   cluster assignments
    centroids_scaled : (K, 3) cluster centroids in scaled space
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)
    c_2d = pca.transform(centroids_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=labels, cmap="Set1", alpha=0.5, s=20, edgecolor="none")
    ax.scatter(c_2d[:, 0], c_2d[:, 1],
               c="black", marker="X", s=200, edgecolors="white", linewidths=1.5,
               zorder=5, label="Centroids")

    var_explained = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC 1 ({var_explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC 2 ({var_explained[1]:.1%} variance)")
    ax.set_title("K-Means Clusters (PCA Projection)")
    cluster_colors = plt.colormaps["Set1"]
    cluster_handles = []
    for cluster_id in sorted(np.unique(labels)):
        cluster_handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                color="w",
                markerfacecolor=cluster_colors(cluster_id / max(best := max(1, len(np.unique(labels)) - 1), 1)),
                markeredgecolor=cluster_colors(cluster_id / max(best, 1)),
                alpha=0.7,
                markersize=8,
                label=f"Cluster {cluster_id}",
            )
        )
    centroid_handle = plt.Line2D(
        [0], [0],
        marker="X",
        linestyle="",
        color="black",
        markersize=10,
        label="Centroids",
    )
    ax.legend(handles=cluster_handles + [centroid_handle], title="Legend", loc="upper right")

    fig.tight_layout()
    return fig


def plot_silhouette_samples_fig(X_scaled, labels, best_k):
    """
    Silhouette samples knife-shape plot: per-sample silhouette values grouped
    by cluster, with a vertical line for the overall average score.
    """
    sample_values = silhouette_samples(X_scaled, labels)
    avg_score = silhouette_score(X_scaled, labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    cmap = plt.colormaps["Set1"]

    for i in range(best_k):
        cluster_vals = sample_values[labels == i]
        cluster_vals.sort()
        y_upper = y_lower + len(cluster_vals)
        color = cmap(i / best_k)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals,
                         facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(cluster_vals), str(i),
                fontsize=12, fontweight="bold")
        y_lower = y_upper + 10

    ax.axvline(x=avg_score, color="red", linestyle="--",
               label=f"Mean silhouette = {avg_score:.3f}")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Sample Index (grouped by cluster)")
    ax.set_title("Silhouette Samples Plot")
    ax.set_yticks([])
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_cluster_vs_smoker(labels, smoker_series):
    """
    Cross-tabulation heatmap: cluster label vs actual smoker status.
    This is the key validation — high diagonal agreement means K-means
    recovers the smoker / non-smoker split without supervision.

    Parameters
    ----------
    labels        : (n,) cluster assignments (0 or 1)
    smoker_series : pd.Series with original string labels ('yes' / 'no')
    """
    ct = pd.crosstab(
        pd.Series(labels, name="Cluster"),
        smoker_series.rename("Smoker"),
    )

    # Ensure consistent ordering: 'no' first, 'yes' second
    if "no" in ct.columns and "yes" in ct.columns:
        ct = ct[["no", "yes"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5,
                ax=ax, cbar_kws={"label": "Count"})
    ax.set_title("Cluster Assignment vs Actual Smoker Status")
    ax.set_ylabel("Cluster")
    ax.set_xlabel("Smoker")
    fig.tight_layout()
    return fig


def plot_cluster_feature_distributions(df_raw, labels):
    """
    Box plots comparing cluster feature distributions for age, bmi, charges.
    Helps interpret *what* each cluster represents.
    """
    tmp = df_raw[CLUSTER_FEATURES].copy()
    tmp["Cluster"] = labels

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, CLUSTER_FEATURES):
        sns.boxplot(data=tmp, x="Cluster", y=col, hue="Cluster", palette="Set1", legend=False, ax=ax)
        ax.set_title(f"{col} by Cluster")

    fig.tight_layout()
    return fig


# Main Interface 


def run_kmeans(path="../data/insurance.csv", best_k=2):
    """
    End-to-end K-means pipeline. Called by page_data_exploration.py.

    Parameters
    ----------
    path   : str — path to raw CSV
    best_k : int — number of clusters (default 2, justified by elbow/silhouette)

    Returns
    -------
    results : dict with keys
        labels          — (n,) cluster assignments
        centroids       — (K, 3) centroids in scaled space
        inertias        — list[float] for K_RANGE
        sil_scores      — list[float] for K_RANGE
        best_sil        — float, silhouette score at best_k
        fig_elbow       — matplotlib Figure
        fig_silhouette  — matplotlib Figure
        fig_pca         — matplotlib Figure
        fig_vs_smoker   — matplotlib Figure
        fig_box         — matplotlib Figure
    """
    # 1. Prepare data
    X_scaled, df_raw, scaler = _prepare_clustering_data(path)

    # 2. Elbow + silhouette sweep
    inertias, sil_scores = _elbow_silhouette(X_scaled)

    # 3. Fit final model
    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    centroids = km.cluster_centers_
    best_sil = silhouette_score(X_scaled, labels)

    # 4. Build all figures
    fig_elbow      = plot_elbow(inertias)
    fig_silhouette = plot_silhouette_scores(sil_scores)
    fig_pca        = plot_clusters_pca(X_scaled, labels, centroids)
    fig_sil_samples = plot_silhouette_samples_fig(X_scaled, labels, best_k)
    fig_vs_smoker  = plot_cluster_vs_smoker(labels, df_raw["smoker"])
    fig_box        = plot_cluster_feature_distributions(df_raw, labels)

    return {
        "labels":         labels,
        "centroids":      centroids,
        "inertias":       inertias,
        "sil_scores":     sil_scores,
        "best_sil":       best_sil,
        "fig_elbow":      fig_elbow,
        "fig_silhouette": fig_silhouette,
        "fig_pca":        fig_pca,
        "fig_vs_smoker":  fig_vs_smoker,
        "fig_box":        fig_box,
        "fig_sil_samples": fig_sil_samples,
    }


# Standalone Test 

if __name__ == "__main__":
    results = run_kmeans()
    print(f"Silhouette score (K=2): {results['best_sil']:.3f}")
    print(f"Cluster sizes: {np.bincount(results['labels'])}")
    plt.show()
