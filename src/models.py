"""Definicje modeli: XGBoost (gradient-boosting) + LSTM (fine-tuning w PyTorch)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from xgboost import XGBRegressor


# --------------------------------------------------------------------------- #
# XGBoost
# --------------------------------------------------------------------------- #
def make_xgb(params: dict | None = None) -> XGBRegressor:
    defaults = dict(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=2,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    if params:
        defaults.update(params)
    return XGBRegressor(**defaults)


# --------------------------------------------------------------------------- #
# LSTM
# --------------------------------------------------------------------------- #
class SequenceDataset(Dataset):
    """Buduje okna `seq_len` z tabeli cech + targetu."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        assert len(X) == len(y)
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx: int):
        x = self.X[idx : idx + self.seq_len]
        target = self.y[idx + self.seq_len - 1]
        return torch.from_numpy(x), torch.tensor(target)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):  # x: (B, T, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# --------------------------------------------------------------------------- #
# Pętla treningowa dla LSTM (fine-tuning)
# --------------------------------------------------------------------------- #
@dataclass
class TrainState:
    best_val: float
    best_state: dict
    history: list


def train_lstm(
    model: LSTMRegressor,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    *,
    epochs: int = 40,
    lr: float = 1e-3,
    patience: int = 6,
    device: str | None = None,
) -> TrainState:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.SmoothL1Loss()

    best = TrainState(best_val=float("inf"), best_state={}, history=[])
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(xb)
            n += len(xb)
        tr_loss /= max(n, 1)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vl, vn = 0.0, 0
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    pred = model(xb)
                    vl += loss_fn(pred, yb).item() * len(xb)
                    vn += len(xb)
                val_loss = vl / max(vn, 1)
            scheduler.step(val_loss)

            if val_loss < best.best_val - 1e-6:
                best.best_val = val_loss
                best.best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    best.history.append((ep, tr_loss, val_loss))
                    break

        best.history.append((ep, tr_loss, val_loss))

    if best.best_state:
        model.load_state_dict(best.best_state)
    return best
