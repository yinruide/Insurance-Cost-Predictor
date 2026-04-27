"""MLP Regressor in PyTorch."""

from pathlib import Path
import sys
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREPROCESS_DIR = ROOT / "preprocess"
SAVED_DIR = ROOT / "saved_models"

sys.path.insert(0, str(PREPROCESS_DIR))
from preprocess import get_regressor_data_torch, split_scaled


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def make_loader(X, y, batch_size=32, shuffle=False):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)

    return total_loss / total_n


def train_mlp(
    epochs=500,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    patience=40,
    seed=42,
):
    set_seed(seed)
    SAVED_DIR.mkdir(exist_ok=True)

    df = get_regressor_data_torch(path=str(ROOT / "data" / "insurance.csv"))
    X_train, X_test, y_train, y_test, scaler = split_scaled(df, "charges")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
    )

    train_loader = make_loader(X_tr, y_tr, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(input_dim=X_train.shape[1]).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val = float("inf")
    best_state = None
    wait = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            seen += xb.size(0)

        train_loss = running_loss / seen
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f}"
            )

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy().reshape(-1)

    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
        "best_val_loss": float(best_val),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    torch.save(best_state, SAVED_DIR / "mlp_regressor.pt")

    with open(SAVED_DIR / "mlp_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nMLP test metrics")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return model, scaler, metrics, history


def fit_mlp():
    """Compatibility wrapper used by the app comparison layer."""
    return train_mlp()


if __name__ == "__main__":
    train_mlp()
