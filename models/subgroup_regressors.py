"""Subgroup-specific regressor helpers for smoker and non-smoker populations."""

from pathlib import Path
import sys
import json

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREPROCESS_DIR = ROOT / "preprocess"
SAVED_DIR = ROOT / "saved_models"

sys.path.insert(0, str(PREPROCESS_DIR))
from preprocess import get_regressor_data_tree, split_unscaled


def _fit_group(df_group):
    X_train, X_test, y_train, y_test = split_unscaled(df_group, "charges")
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=12138,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
    }
    return model, metrics


def fit_subgroup_regressors():
    """Train and score smoker and non-smoker regressors separately."""
    df = get_regressor_data_tree(path=str(ROOT / "data" / "insurance.csv"))
    smoker_model, smoker_metrics = _fit_group(df[df["smoker"] == 1])
    nonsmoker_model, nonsmoker_metrics = _fit_group(df[df["smoker"] == 0])

    SAVED_DIR.mkdir(exist_ok=True)
    with open(SAVED_DIR / "subgroup_regressor_metrics.json", "w") as f:
        json.dump(
            {
                "smoker": smoker_metrics,
                "nonsmoker": nonsmoker_metrics,
            },
            f,
            indent=2,
        )

    return smoker_model, nonsmoker_model, smoker_metrics, nonsmoker_metrics


if __name__ == "__main__":
    _, _, smoker_metrics, nonsmoker_metrics = fit_subgroup_regressors()
    print({"smoker": smoker_metrics, "nonsmoker": nonsmoker_metrics})