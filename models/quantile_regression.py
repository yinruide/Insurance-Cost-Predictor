"""Quantile Regression with pinball loss."""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREPROCESS_DIR = ROOT / "preprocess"
SAVED_DIR = ROOT / "saved_models"

sys.path.insert(0, str(PREPROCESS_DIR))
from preprocess import get_regressor_data_linear, split_scaled


def fit_quantile_models(quantiles=(0.1, 0.5, 0.9), alpha=0.0):
    SAVED_DIR.mkdir(exist_ok=True)

    df = get_regressor_data_linear(path=str(ROOT / "data" / "insurance.csv"))
    X_train, X_test, y_train, y_test, scaler = split_scaled(df, "log_charges")

    models = {}
    pred_list = []

    for q in quantiles:
        model = QuantileRegressor(
            quantile=q,
            alpha=alpha,
            solver="highs",
        )
        model.fit(X_train, y_train)
        models[q] = model
        pred_list.append(model.predict(X_test))

    preds_log = np.column_stack(pred_list)

    # avoid quantile crossing
    preds_log.sort(axis=1)

    lower_log = preds_log[:, 0]
    median_log = preds_log[:, 1]
    upper_log = preds_log[:, 2]

    y_test_dollar = np.expm1(y_test)
    lower_dollar = np.expm1(lower_log)
    median_dollar = np.expm1(median_log)
    upper_dollar = np.expm1(upper_log)

    coverage = np.mean(
        (y_test_dollar >= lower_dollar) &
        (y_test_dollar <= upper_dollar)
    )

    avg_interval_width = np.mean(upper_dollar - lower_dollar)

    metrics = {
        "mae_median_dollar": float(mean_absolute_error(y_test_dollar, median_dollar)),
        "rmse_median_dollar": float(np.sqrt(mean_squared_error(y_test_dollar, median_dollar))),
        "r2_median_dollar": float(r2_score(y_test_dollar, median_dollar)),
        "interval_80_coverage": float(coverage),
        "avg_interval_width_dollar": float(avg_interval_width),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    pred_df = pd.DataFrame({
        "actual_charges": y_test_dollar,
        "q10": lower_dollar,
        "q50": median_dollar,
        "q90": upper_dollar,
    })

    pred_df.to_csv(SAVED_DIR / "quantile_predictions.csv", index=False)

    with open(SAVED_DIR / "quantile_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nQuantile regression test metrics")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return models, scaler, metrics, pred_df


if __name__ == "__main__":
    fit_quantile_models()