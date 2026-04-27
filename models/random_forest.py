"""Random Forest regressor for insurance cost prediction."""

from pathlib import Path
import sys
import json

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREPROCESS_DIR = ROOT / "preprocess"
SAVED_DIR = ROOT / "saved_models"

sys.path.insert(0, str(PREPROCESS_DIR))
from preprocess import get_regressor_data_tree, split_unscaled


def fit_random_forest():
    """Train a random forest regressor and persist metrics plus model weights."""
    SAVED_DIR.mkdir(exist_ok=True)

    df = get_regressor_data_tree(path=str(ROOT / "data" / "insurance.csv"))
    X_train, X_test, y_train, y_test = split_unscaled(df, "charges")

    model = RandomForestRegressor(
        n_estimators=500,
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
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    joblib.dump(model, SAVED_DIR / "rf_regressor_full.pkl")

    with open(SAVED_DIR / "random_forest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nRandom forest test metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return model, metrics


if __name__ == "__main__":
    fit_random_forest()
