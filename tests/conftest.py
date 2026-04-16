"""Syntetyczne dane na potrzeby testów — bez pobierania z Kaggle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_raw(tmp_path: Path) -> pd.DataFrame:
    """Ramka z kolumnami takimi jak prawdziwy CSV z Kaggle."""
    rng = np.random.default_rng(42)
    n = 1200
    dates = pd.bdate_range("2015-01-01", periods=n)

    # losowe błądzenia z dryfem — imitacja cen
    gold = 1200 + np.cumsum(rng.normal(0.05, 12, n))
    silver = 16 + np.cumsum(rng.normal(0.001, 0.25, n))
    gprd = np.abs(rng.normal(90, 35, n)) + 20 * np.sin(np.arange(n) / 60)

    return pd.DataFrame(
        {
            "DATE": dates,
            "GOLD_PRICE": gold,
            "SILVER_PRICE": silver,
            "GPRD": gprd,
            "GPRD_ACT": gprd * 0.6,
            "GPRD_THREAT": gprd * 0.4,
        }
    )


@pytest.fixture
def cfg(tmp_path: Path, synthetic_raw: pd.DataFrame) -> dict:
    """Minimalny config wskazujący na syntetyczny CSV w tmp_path."""
    csv_path = tmp_path / "synthetic.csv"
    synthetic_raw.to_csv(csv_path, index=False)
    return {
        "paths": {
            "raw_csv": str(csv_path),
            "processed_pickle": str(tmp_path / "processed.pkl"),
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "columns": {
            "date_candidates": ["DATE", "Date"],
            "gold_candidates": ["GOLD_PRICE"],
            "silver_candidates": ["SILVER_PRICE"],
            "gpr_candidates": ["GPRD", "GPR"],
            "gpr_act_candidates": ["GPRD_ACT"],
            "gpr_threat_candidates": ["GPRD_THREAT"],
        },
        "features": {
            "lags": [1, 5, 20],
            "rolling_windows": [5, 20],
            "use_returns": True,
            "use_log_prices": True,
        },
        "split": {"test_size_months": 6, "n_cv_splits": 3},
        "target": {"asset": "gold", "horizon_days": 20, "mode": "return"},
        "tuning": {"n_trials": 2, "timeout_sec": 30, "direction": "minimize", "seed": 42},
        "lstm": {
            "seq_len": 20,
            "hidden": 16,
            "layers": 1,
            "dropout": 0.1,
            "epochs": 2,
            "batch_size": 32,
            "lr": 1e-3,
            "patience": 2,
        },
    }
