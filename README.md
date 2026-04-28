# Medical Insurance Cost Predictor

A Streamlit web app that predicts annual medical insurance costs using ensemble methods, neural networks, and a from-scratch Mixture Density Network.

**Team:** Gentle Eagles  (Ruide Yin, Ben Ronen, Allison Zhu, Ruize Ma)   
**Course:** Spring 2026, Fundamentals of Machine Learning, Final Project, NYU     
**Dataset:** [Kaggle Medical Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) — 1,338 records, 7 features, no missing values

---

## Overview

Medical costs are opaque and difficult to anticipate. This app lets a user input their health and demographic profile and receive an interpretable, data-driven estimate of their annual insurance charges — including an 80% uncertainty band and an explanation of which model path was used.

The charge distribution is strongly bimodal (driven by smoking status), which motivates a richer modeling approach than a single regression. We address this with two parallel modeling blocks and a three-page interactive Streamlit interface.

## Model Architecture

### Block 1 — Direct Regression

| Model | Description |
|---|---|
| Linear Regression | Baseline model on log-transformed charges |
| Random Forest | GridSearchCV-tuned ensemble (best params: max_depth=10, n_estimators=300) |
| XGBoost | GridSearchCV-tuned gradient boosting |
| MLP Regressor | PyTorch feed-forward network (64→32→16→1) with early stopping |
| Quantile Regression | Pinball loss at quantiles 0.1, 0.5, 0.9 for prediction intervals |

### Block 2 — Stratified Pipeline

A binary classifier estimates smoker probability from demographics, then routes the prediction to smoker-specific or non-smoker-specific Random Forest regressors. When smoker status is unknown, the final estimate blends both subgroup predictions proportionally.

### From-Scratch Algorithm — Mixture Density Network (MDN)

The MDN is implemented from scratch in PyTorch (no library wrapper), satisfying the course rubric requirement. It models the bimodal charge distribution as a mixture of Gaussians, learning per-input mixing coefficients, means, and variances. The final model uses 2 Gaussian components, directly corresponding to the smoker/non-smoker cost modes identified in EDA.

### Unsupervised Validation — K-Means Clustering

K-means clustering (K=2–8, elbow + silhouette analysis) validates the EDA finding that smoker/non-smoker populations form naturally separable clusters in feature space, providing unsupervised evidence for the Block 2 stratified pipeline.

## Results

All metrics are evaluated on the same 20% held-out test set (n=268, `random_state=12138`, stratified by smoker).

| Model | R² | RMSE ($) | MAE ($) |
|---|---|---|---|
| XGBoost | 0.846 | 4,759 | 2,602 |
| Random Forest | 0.843 | 4,805 | 2,630 |
| MLP | 0.810 | 5,281 | 3,141 |
| MDN | 0.790 | 5,552 | 2,874 |
| Linear Regression | 0.587 | 7,784 | 4,066 |
| Quantile Regression | 0.538 | 8,234 | 4,109 |

Quantile Regression achieves 78.4% empirical coverage on the 80% prediction interval (q10–q90), with an average interval width of $9,223. Its value is in uncertainty quantification rather than point accuracy — the intervals power the confidence band shown on the live predictor page.

The Block 2 stratified pipeline uses a logistic regression classifier (F1=0.342 on test) to estimate smoker probability from demographics alone — a deliberately modest score reflecting that demographics are weak predictors of smoking status. The subgroup regressors achieve smoker R²=0.833 and non-smoker R²=0.469, confirming that separating the two populations improves smoker predictions substantially.

## Streamlit App

The app has three pages:

**Page 1 — Data Exploration:** Interactive EDA with sidebar filters, tabbed visualizations (feature distributions, feature-vs-charges relationships, correlations), and a K-means clustering section with a dynamic K slider.

**Page 2 — Live Cost Predictor:** Enter a profile to get a predicted annual cost, an 80% uncertainty band (from quantile regression), routing explanation (which model path was used), and a feature impact chart showing what drove the estimate.

**Page 3 — Model Comparison:** Side-by-side leaderboard of all models on held-out test metrics, R² bar chart, and quantile regression interval diagnostics.

## Repository Structure

```
Insurance-Cost-Predictor/
├── data/
│   └── insurance.csv
├── exploration/
│   ├── eda.ipynb               # Full EDA with statistical tests
│   ├── eda_utils.py            # 12 reusable plotting functions
│   ├── block2_eda.ipynb        # Smoker-stratified EDA for Block 2
│   └── kmeans.py               # K-means pipeline (K=2–8, 5 visualizations)
├── preprocess/
│   └── preprocess.py           # Encoding, feature engineering, train/test split, scaling
├── models/
│   ├── linear_regression.py    # Baseline linear model
│   ├── random_forest.py        # RF with GridSearchCV
│   ├── xgboost_model.py        # XGBoost with GridSearchCV
│   ├── mlp.py                  # MLP regressor (PyTorch)
│   ├── mdn.py                  # Mixture Density Network (from scratch, PyTorch)
│   ├── quantile_regression.py  # Quantile regression (q10, q50, q90)
│   ├── block2_classifier.py    # Binary smoker classifier + subgroup regressors
│   ├── smoker_classifier.py    # Classifier utilities for live inference
│   └── subgroup_regressors.py  # Subgroup regressor utilities
├── evaluation/
│   └── feature_importance.py   # Feature importance helpers
├── app/
│   ├── app.py                  # Multi-page Streamlit entry point
│   ├── shared.py               # Design system, UI components, inference logic
│   ├── page_data_exploration.py
│   ├── page_cost_predictor.py
│   └── page_model_comparison.py
├── saved_models/               # Trained model artifacts and metric JSONs
├── requirements.txt
└── README.md
```

## Setup and Usage

```bash
# Clone the repository
git clone https://github.com/ry2406/Insurance-Cost-Predictor.git
cd Insurance-Cost-Predictor

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate             # Windows
# source .venv/bin/activate        # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app/app.py
```

Pre-trained model artifacts are included in `saved_models/`. The app loads them directly on startup — no retraining needed. If any artifact is missing, the app will automatically retrain that model on first launch.

To retrain a specific model manually:

```bash
python models/linear_regression.py
python models/random_forest.py
python models/xgboost_model.py
python models/mlp.py
python models/quantile_regression.py
python models/block2_classifier.py
```

## Key Design Decisions

- **EDA before preprocessing:** All preprocessing choices (log transforms, outlier treatment, encoding) are grounded in EDA findings, not assumed upfront.
- **Log-transform scoping:** `log1p(charges)` is applied only to models assuming normality (Linear Regression, Quantile Regression). Neural models (MLP, MDN) train on raw charges.
- **Data leakage prevention:** Scaling is performed after train/test split, fit only on training data. The `bmi_smoker` interaction feature is excluded from classifier training to avoid target leakage.
- **Consistent evaluation:** All models share the same 80/20 split with `random_state=12138`, stratified by smoker status.

## Tech Stack

Python, PyTorch, scikit-learn, XGBoost, Streamlit, pandas, NumPy, matplotlib, seaborn

## Team Contributions

| Member | Actual Contributions |
|---|---|
| **Ruide Yin** | Data preprocessing pipeline (`preprocess.py`, `PREPROCESS_GUIDE.md`), full EDA (`eda.ipynb`, `eda_utils.py`), K-means clustering (`kmeans.py`), Streamlit Page 1 (`page_data_exploration.py`), repository architecture design, code review and technical coordination across all teammates, final branch merge and integration |
| **Ben Ronen** | Block 2 binary smoker classifier and subgroup regressors (`block2_classifier.py`, `block2_eda.ipynb`), Mixture Density Network from scratch (`mdn.py`), full-stack app integration: design system and inference pipeline (`shared.py`), Streamlit Page 2 (`page_cost_predictor.py`), Page 3 refactor (`page_model_comparison.py`), app shell (`app.py`), runtime stability fixes, model utility modules (`linear_regression.py`, `smoker_classifier.py`, `subgroup_regressors.py`, `feature_importance.py`) |
| **Allison Zhu** | Random Forest with GridSearchCV (`random_forest.py`), XGBoost with GridSearchCV (`xgboost_model.py`), initial cost predictor page draft |
| **Ruize Ma** | MLP regressor in PyTorch (`mlp.py`), Quantile Regression (`quantile_regression.py`), app shell initial draft (`app.py`), model comparison page initial draft (`page_model_comparison.py`) |
