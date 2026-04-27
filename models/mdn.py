"""Mixture Density Network implemented from scratch in PyTorch."""

from pathlib import Path
import sys
import json
import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREPROCESS_DIR = ROOT / "preprocess"
SAVED_DIR = ROOT / "saved_models"

sys.path.insert(0, str(PREPROCESS_DIR))
from preprocess import get_regressor_data_mdn


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MixtureDensityNetwork(nn.Module):
    """Two-layer MDN head over a small MLP encoder."""

    def __init__(self, input_dim, hidden_dim=64, n_components=2):
        super().__init__()
        self.n_components = n_components
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pi_head = nn.Linear(hidden_dim, n_components)
        self.mu_head = nn.Linear(hidden_dim, n_components)
        self.sigma_head = nn.Linear(hidden_dim, n_components)

    def forward(self, x):
        h = self.backbone(x)
        pi = torch.softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h)
        sigma = torch.exp(self.sigma_head(h)) + 1e-4
        return pi, mu, sigma


def mdn_negative_log_likelihood(pi, mu, sigma, target):
    """Gaussian mixture negative log likelihood."""
    target = target.expand_as(mu)
    log_coeff = -torch.log(sigma) - 0.5 * math.log(2.0 * math.pi)
    log_exp = -0.5 * ((target - mu) / sigma) ** 2
    component_log_probs = log_coeff + log_exp
    mixture_log_probs = torch.log(pi + 1e-8) + component_log_probs
    return -torch.logsumexp(mixture_log_probs, dim=1).mean()


def mixture_mean(pi, mu):
    """Expected value of the Gaussian mixture."""
    return torch.sum(pi * mu, dim=1)


def fit_mdn(
    epochs=400,
    batch_size=64,
    lr=1e-3,
    patience=40,
    seed=42,
    n_components=2,
):
    """Train the MDN, save weights/metrics, and return predictions."""
    set_seed(seed)
    SAVED_DIR.mkdir(exist_ok=True)

    df = get_regressor_data_mdn(path=str(ROOT / "data" / "insurance.csv")).copy()
    feature_cols = [
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "bmi_smoker",
        "region_northwest",
        "region_southeast",
        "region_southwest",
    ]
    X = df[feature_cols].values
    y = np.log1p(df["charges"].values)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=12138,
        stratify=df["smoker"],
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MixtureDensityNetwork(
        input_dim=X_train.shape[1],
        hidden_dim=64,
        n_components=n_components,
    ).to(device)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).view(-1, 1),
    )
    val_x = torch.tensor(X_val, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pi, mu, sigma = model(xb)
            loss = mdn_negative_log_likelihood(pi, mu, sigma, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pi_val, mu_val, sigma_val = model(val_x)
            val_loss = mdn_negative_log_likelihood(pi_val, mu_val, sigma_val, val_y).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 25 == 0:
            print(f"Epoch {epoch:03d} | val_nll={val_loss:.4f}")

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        pi_test, mu_test, sigma_test = model(X_test_tensor)
        pred_log = mixture_mean(pi_test, mu_test).cpu().numpy()

    y_test_dollar = np.expm1(y_test)
    pred_dollar = np.expm1(pred_log)

    metrics = {
        "mae": float(mean_absolute_error(y_test_dollar, pred_dollar)),
        "rmse": float(np.sqrt(mean_squared_error(y_test_dollar, pred_dollar))),
        "r2": float(r2_score(y_test_dollar, pred_dollar)),
        "best_val_nll": float(best_val),
        "n_components": int(n_components),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    torch.save(
        {
            "state_dict": best_state,
            "input_dim": int(X_train.shape[1]),
            "n_components": int(n_components),
            "feature_cols": feature_cols,
        },
        SAVED_DIR / "mdn_regressor.pt",
    )

    with open(SAVED_DIR / "mdn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nMDN test metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    return model, scaler, metrics


if __name__ == "__main__":
    fit_mdn()
