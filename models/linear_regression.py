"""Linear Regression baseline for insurance cost prediction."""

from pathlib import Path
import sys
import json

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREPROCESS_DIR = ROOT / "preprocess"
SAVED_DIR = ROOT / "saved_models"

sys.path.insert(0, str(PREPROCESS_DIR))
from preprocess import get_regressor_data_linear, split_scaled


def fit_linear_regression():
    """Train a linear baseline on log-charges and save evaluation metrics."""
    SAVED_DIR.mkdir(exist_ok=True)

    df = get_regressor_data_linear(path=str(ROOT / "data" / "insurance.csv"))
    X_train, X_test, y_train, y_test, scaler = split_scaled(df, "log_charges")

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_log = model.predict(X_test)
    y_test_dollar = np.expm1(y_test)
    pred_dollar = np.expm1(pred_log)

    metrics = {
        "mae_dollar": float(mean_absolute_error(y_test_dollar, pred_dollar)),
        "rmse_dollar": float(np.sqrt(mean_squared_error(y_test_dollar, pred_dollar))),
        "r2_dollar": float(r2_score(y_test_dollar, pred_dollar)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    with open(SAVED_DIR / "linear_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nLinear regression test metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return model, scaler, metrics


if __name__ == "__main__":
    fit_linear_regression()
