"""
block2_classifier.py
Block 2 Pipeline: Smoker Classifier + Stratified Subgroup Regressors

Pipeline:
  1. Binary smoker classifier (Logistic Regression baseline, Random Forest main)
  2. Smoker subgroup regressor (Random Forest on charges)
  3. Non-smoker subgroup regressor (Random Forest on charges)

EDA motivation (block2_eda.ipynb):
  - Smoker/non-smoker charge distributions are near-disjoint ($32k vs $8k mean),
    so routing predictions through a classifier first materially reduces regressor error.
  - The smoker KDE is bimodal (~$20k and ~$40k peaks), likely BMI-driven —
    a tree-based regressor handles this better than a linear model.
"""

import sys
import os
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, r2_score,
)

# Allow imports from sibling preprocess/ directory when run from models/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "preprocess"))
from preprocess import (
    get_classifier_data_logistic,
    get_classifier_data_tree,
    get_regressor_data_tree,
    split_scaled,
    split_unscaled,
)

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "insurance.csv")


# ── Classifier helpers ────────────────────────────────────────────────────────

def train_logistic_classifier(X_train, y_train):
    """Logistic Regression baseline classifier. class_weight='balanced' corrects the ~20/80 smoker split."""
    model = LogisticRegression(max_iter=1000, random_state=12138, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def train_rf_classifier(X_train, y_train):
    """
    Random Forest classifier (main model).
    class_weight='balanced_subsample' re-weights per bootstrap draw — more effective
    than 'balanced' for imbalanced classes in tree ensembles.
    Shallow max_depth prevents majority-class over-fitting.
    """
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=12138,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, X_test, y_test, name="Model"):
    """Print accuracy, precision, recall, F1, and confusion matrix."""
    preds = model.predict(X_test)

    acc  = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec  = recall_score(y_test, preds)
    f1   = f1_score(y_test, preds)
    cm   = confusion_matrix(y_test, preds)

    print(f"\n{'─' * 45}")
    print(f"  {name}")
    print(f"{'─' * 45}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Confusion matrix (rows=actual, cols=predicted):")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def save_model(model, filename):
    """Persist a fitted model to saved_models/<filename>."""
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    path = os.path.join(SAVED_MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"  Saved → {path}")


# ── Regressor helpers ─────────────────────────────────────────────────────────

def train_rf_regressor(X_train, y_train):
    """Random Forest regressor for a single subgroup."""
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=12138,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_regressor(model, X_test, y_test, name="Model"):
    """Print RMSE and R² for a regressor."""
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)

    print(f"\n{'─' * 45}")
    print(f"  {name}")
    print(f"{'─' * 45}")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  R²   : {r2:.4f}")

    return {"rmse": rmse, "r2": r2}


# ── Block 2 entry point ───────────────────────────────────────────────────────

def run_block2():
    """
    Full Block 2 pipeline:
      1. Train and compare classifier models; save the best one.
      2. Train smoker and non-smoker subgroup regressors; save both.
    """

    # ── Part 1: Binary Smoker Classifier ─────────────────────────────────────
    print("\n" + "═" * 45)
    print("  PART 1 — Binary Smoker Classifier")
    print("═" * 45)

    # Logistic Regression uses scaled one-hot features
    df_logistic = get_classifier_data_logistic(path=DATA_PATH)
    X_tr_lr, X_te_lr, y_tr_lr, y_te_lr, scaler_lr = split_scaled(df_logistic, "smoker")

    lr_model  = train_logistic_classifier(X_tr_lr, y_tr_lr)
    lr_scores = evaluate_classifier(lr_model, X_te_lr, y_te_lr, name="Logistic Regression (baseline)")

    # Random Forest uses unscaled label-encoded features
    df_tree = get_classifier_data_tree(path=DATA_PATH)
    X_tr_rf, X_te_rf, y_tr_rf, y_te_rf = split_unscaled(df_tree, "smoker")

    rf_model  = train_rf_classifier(X_tr_rf, y_tr_rf)
    rf_scores = evaluate_classifier(rf_model, X_te_rf, y_te_rf, name="Random Forest Classifier (main)")

    # Save the model with the higher F1 score
    if rf_scores["f1"] >= lr_scores["f1"]:
        best_clf, best_name = rf_model, "rf_smoker_classifier.pkl"
        print("\n  Best classifier: Random Forest (by F1)")
    else:
        best_clf, best_name = lr_model, "lr_smoker_classifier.pkl"
        print("\n  Best classifier: Logistic Regression (by F1)")

    save_model(best_clf, best_name)

    # ── Part 2: Stratified Subgroup Regressors ────────────────────────────────
    print("\n" + "═" * 45)
    print("  PART 2 — Stratified Subgroup Regressors")
    print("═" * 45)

    # Load tree-encoded regression data (keeps smoker column for filtering)
    df_reg = get_regressor_data_tree(path=DATA_PATH)

    # Smoker subgroup (smoker == 1)
    s_df = df_reg[df_reg["smoker"] == 1]
    X_tr_s, X_te_s, y_tr_s, y_te_s = split_unscaled(s_df, "charges")
    rf_smoker  = train_rf_regressor(X_tr_s, y_tr_s)
    evaluate_regressor(rf_smoker, X_te_s, y_te_s, name="RF Regressor — Smokers")
    save_model(rf_smoker, "rf_regressor_smoker.pkl")

    # Non-smoker subgroup (smoker == 0)
    n_df = df_reg[df_reg["smoker"] == 0]
    X_tr_n, X_te_n, y_tr_n, y_te_n = split_unscaled(n_df, "charges")
    rf_nonsmoker  = train_rf_regressor(X_tr_n, y_tr_n)
    evaluate_regressor(rf_nonsmoker, X_te_n, y_te_n, name="RF Regressor — Non-Smokers")
    save_model(rf_nonsmoker, "rf_regressor_nonsmoker.pkl")

    print("\n" + "═" * 45)
    print("  Block 2 complete.")
    print("═" * 45 + "\n")


if __name__ == "__main__":
    run_block2()
