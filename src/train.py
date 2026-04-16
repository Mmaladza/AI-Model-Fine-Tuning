"""Trening + fine-tuning hiperparametrów (Optuna) dla XGBoost i LSTM.

Użycie z CLI:
    python -m src.train --model xgb
    python -m src.train --model lstm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.data import load_config, load_processed
from src.features import build_features, split_xy
from src.models import (
    LSTMRegressor,
    SequenceDataset,
    make_xgb,
    train_lstm,
)


# --------------------------------------------------------------------------- #
# Pomocnicze
# --------------------------------------------------------------------------- #
def train_test_split_time(df: pd.DataFrame, months: int):
    cutoff = df["date"].max() - pd.DateOffset(months=months)
    return df[df["date"] <= cutoff].copy(), df[df["date"] > cutoff].copy()


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "hit_rate": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


# --------------------------------------------------------------------------- #
# Optuna — XGBoost
# --------------------------------------------------------------------------- #
def tune_xgb(X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict:
    tscv = TimeSeriesSplit(n_splits=cfg["split"]["n_cv_splits"])

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 9),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        )
        scores = []
        for tr_idx, va_idx in tscv.split(X):
            m = make_xgb(params)
            m.fit(
                X.iloc[tr_idx],
                y.iloc[tr_idx],
                eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
                verbose=False,
            )
            pred = m.predict(X.iloc[va_idx])
            scores.append(mean_absolute_error(y.iloc[va_idx], pred))
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=cfg["tuning"]["seed"])
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=cfg["tuning"]["n_trials"],
        timeout=cfg["tuning"]["timeout_sec"],
        show_progress_bar=True,
    )
    return study.best_params


# --------------------------------------------------------------------------- #
# Optuna — LSTM
# --------------------------------------------------------------------------- #
def tune_lstm(X_tr, y_tr, X_va, y_va, n_features, cfg: dict) -> dict:
    def objective(trial: optuna.Trial) -> float:
        params = dict(
            hidden=trial.suggest_categorical("hidden", [32, 64, 128]),
            layers=trial.suggest_int("layers", 1, 3),
            dropout=trial.suggest_float("dropout", 0.0, 0.4),
            lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            seq_len=trial.suggest_categorical("seq_len", [20, 40, 60, 90]),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        )
        tr_ds = SequenceDataset(X_tr, y_tr, params["seq_len"])
        va_ds = SequenceDataset(X_va, y_va, params["seq_len"])
        if len(tr_ds) < 100 or len(va_ds) < 20:
            raise optuna.TrialPruned()
        tr_ld = DataLoader(tr_ds, batch_size=params["batch_size"], shuffle=True)
        va_ld = DataLoader(va_ds, batch_size=params["batch_size"])

        model = LSTMRegressor(n_features, params["hidden"], params["layers"], params["dropout"])
        state = train_lstm(
            model,
            tr_ld,
            va_ld,
            epochs=min(20, cfg["lstm"]["epochs"]),
            lr=params["lr"],
            patience=3,
        )
        return state.best_val

    sampler = optuna.samplers.TPESampler(seed=cfg["tuning"]["seed"])
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=max(20, cfg["tuning"]["n_trials"] // 2),
        timeout=cfg["tuning"]["timeout_sec"],
        show_progress_bar=True,
    )
    return study.best_params


# --------------------------------------------------------------------------- #
# Główne ścieżki
# --------------------------------------------------------------------------- #
def run_xgb(cfg: dict) -> dict:
    df = load_processed(cfg)
    feats = build_features(df, cfg)
    train_df, test_df = train_test_split_time(feats, cfg["split"]["test_size_months"])

    X_tr, y_tr = split_xy(train_df)
    X_te, y_te = split_xy(test_df)

    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    print("== Fine-tuning XGBoost (Optuna) ==")
    best_params = tune_xgb(X_tr, y_tr, cfg)
    print("Najlepsze parametry:", best_params)

    model = make_xgb(best_params)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    pred_te = model.predict(X_te)
    metrics = evaluate(y_te.values, pred_te)
    print("Metryki (test):", metrics)

    out_dir = Path(cfg["paths"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "features": list(X_tr.columns), "params": best_params, "metrics": metrics},
        out_dir / "xgb.joblib",
    )

    rep_dir = Path(cfg["paths"].get("reports_dir", "reports"))
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "xgb_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def run_lstm(cfg: dict) -> dict:
    df = load_processed(cfg)
    feats = build_features(df, cfg)
    train_df, test_df = train_test_split_time(feats, cfg["split"]["test_size_months"])

    X_tr_df, y_tr = split_xy(train_df)
    X_te_df, y_te = split_xy(test_df)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_df.values)
    X_te = scaler.transform(X_te_df.values)

    # walidacja wewnątrz zbioru treningowego (ostatnie 20%)
    n_val = max(200, int(0.2 * len(X_tr)))
    X_tr_in, X_va = X_tr[:-n_val], X_tr[-n_val:]
    y_tr_in, y_va = y_tr.values[:-n_val], y_tr.values[-n_val:]

    print(f"Train: {X_tr_in.shape}, Val: {X_va.shape}, Test: {X_te.shape}")
    print("== Fine-tuning LSTM (Optuna) ==")
    best = tune_lstm(X_tr_in, y_tr_in, X_va, y_va, X_tr.shape[1], cfg)
    print("Najlepsze parametry:", best)

    # finalny trening na train + val z najlepszymi parametrami
    seq_len = best["seq_len"]
    tr_ds = SequenceDataset(X_tr, y_tr.values, seq_len)
    te_ds = SequenceDataset(X_te, y_te.values, seq_len)
    tr_ld = DataLoader(tr_ds, batch_size=best["batch_size"], shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=best["batch_size"])

    model = LSTMRegressor(X_tr.shape[1], best["hidden"], best["layers"], best["dropout"])
    train_lstm(
        model,
        tr_ld,
        te_ld,
        epochs=cfg["lstm"]["epochs"],
        lr=best["lr"],
        patience=cfg["lstm"]["patience"],
    )

    # ocena na teście (okna)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    preds, truth = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            preds.append(model(xb.to(device)).cpu().numpy())
            truth.append(yb.numpy())
    preds = np.concatenate(preds)
    truth = np.concatenate(truth)
    metrics = evaluate(truth, preds)
    print("Metryki (test):", metrics)

    out_dir = Path(cfg["paths"]["models_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "params": best,
            "n_features": X_tr.shape[1],
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "feature_names": list(X_tr_df.columns),
            "metrics": metrics,
        },
        out_dir / "lstm.pt",
    )

    rep_dir = Path(cfg["paths"].get("reports_dir", "reports"))
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "lstm_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgb", "lstm", "both"], default="xgb")
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.model in ("xgb", "both"):
        run_xgb(cfg)
    if args.model in ("lstm", "both"):
        run_lstm(cfg)


if __name__ == "__main__":
    main()
