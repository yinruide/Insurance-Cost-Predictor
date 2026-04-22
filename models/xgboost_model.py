"""XGBoost with GridSearchCV."""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import os

def load_data(path="data/insurance.csv"):
    df = pd.read_csv(path)
    le = LabelEncoder()
    df["sex"]    = le.fit_transform(df["sex"])
    df["smoker"] = le.fit_transform(df["smoker"])
    df["region"] = le.fit_transform(df["region"])
    return df

def get_splits(df):
    X = df.drop("charges", axis=1)
    y = df["charges"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def tune_xgboost(X_train, y_train):
    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample":     [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }
    xgb = XGBRegressor(random_state=42, verbosity=0)
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
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

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = get_splits(df)

    print("Tuning XGBoost...")
    best_model, best_params = tune_xgboost(X_train, y_train)

    print("\nEvaluation on test set:")
    metrics = evaluate(best_model, X_test, y_test)

    feature_importance = get_feature_importance(best_model, X_train.columns.tolist())
    print(f"\nFeature importances: {feature_importance}")

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(best_model, "saved_models/xgboost_model.pkl")

    results = {"best_params": best_params, "metrics": metrics, "feature_importance": feature_importance}
    with open("saved_models/xgb_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved to saved_models/")


