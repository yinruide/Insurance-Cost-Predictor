"""Binary smoker classifier utilities."""

from pathlib import Path
import sys
import json

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREPROCESS_DIR = ROOT / "preprocess"
SAVED_DIR = ROOT / "saved_models"

sys.path.insert(0, str(PREPROCESS_DIR))
from preprocess import get_classifier_data_logistic, split_scaled


def fit_smoker_classifier():
    """Train the logistic smoker classifier and save its evaluation metrics."""
    df = get_classifier_data_logistic(path=str(ROOT / "data" / "insurance.csv"))
    X_train, X_test, y_train, y_test, scaler = split_scaled(df, "smoker")

    model = LogisticRegression(
        max_iter=1000,
        random_state=12138,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
    }

    SAVED_DIR.mkdir(exist_ok=True)
    with open(SAVED_DIR / "smoker_classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, scaler, metrics


if __name__ == "__main__":
    _, _, metrics = fit_smoker_classifier()
    print(metrics)