"""XGBoost with GridSearchCV."""

import sys
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREPROCESS_DIR = ROOT / "preprocess"
SAVED_DIR = ROOT / "saved_models"
sys.path.insert(0, str(PREPROCESS_DIR))

from preprocess import get_regressor_data_tree, split_unscaled

def tune_xgboost(X_train, y_train):
    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample":     [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }
    xgb = XGBRegressor(random_state=12138, verbosity=0)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)
    print(f"Best params: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "MAE":  round(mean_absolute_error(y_test, preds), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 2),
        "R2":   round(r2_score(y_test, preds), 4),
    }
    print(f"MAE:  {metrics['MAE']}")
    print(f"RMSE: {metrics['RMSE']}")
    print(f"R2:   {metrics['R2']}")
    return metrics

def get_feature_importance(model, feature_names):
    importances = model.feature_importances_
    return dict(zip(feature_names, [round(float(v), 4) for v in importances]))

def fit_xgboost():
    df = get_regressor_data_tree(path=str(ROOT / "data" / "insurance.csv"))
    feature_names = [c for c in df.columns if c not in {"charges", "log_charges", "smoker"}]
    X_train, X_test, y_train, y_test = split_unscaled(df, "charges")

    print("Tuning XGBoost...")
    best_model, best_params = tune_xgboost(X_train, y_train)

    print("\nEvaluation on test set:")
    metrics = evaluate(best_model, X_test, y_test)

    feature_importance = get_feature_importance(best_model, feature_names)
    print(f"\nFeature importances: {feature_importance}")

    SAVED_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, SAVED_DIR / "xgboost_model.pkl")

    results = {"best_params": best_params, "metrics": metrics, "feature_importance": feature_importance}
    with open(SAVED_DIR / "xgb_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to saved_models/")
    return results

if __name__ == "__main__":
    fit_xgboost()
