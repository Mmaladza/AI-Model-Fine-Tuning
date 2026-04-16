"""Smoke-testy modeli — sprawdzają, czy się inicjują i uczą na małym zbiorze."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models import LSTMRegressor, SequenceDataset, make_xgb, train_lstm


def test_make_xgb_defaults_are_sensible():
    m = make_xgb()
    # wartości kluczowe dla szeregów czasowych: regularizacja i hist tree
    assert m.get_params()["tree_method"] == "hist"
    assert m.get_params()["subsample"] <= 1.0


def test_sequence_dataset_shapes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 5)).astype(np.float32)
    y = rng.standard_normal(100).astype(np.float32)
    ds = SequenceDataset(X, y, seq_len=10)
    x0, y0 = ds[0]
    assert x0.shape == (10, 5)
    assert y0.shape == ()
    assert len(ds) == 90


def test_lstm_trains_and_reduces_loss():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4)).astype(np.float32)
    y = X[:, 0] * 0.5 + rng.standard_normal(200) * 0.01  # cel silnie zależny od X[:,0]
    y = y.astype(np.float32)

    ds = SequenceDataset(X, y, seq_len=10)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    model = LSTMRegressor(n_features=4, hidden=8, layers=1, dropout=0.0)

    state = train_lstm(model, loader, val_loader=loader, epochs=3, lr=1e-2, patience=3)
    losses = [h[1] for h in state.history]
    assert losses[-1] < losses[0]  # jakikolwiek uczący się model
