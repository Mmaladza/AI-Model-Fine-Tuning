"""Inżynieria cech: lagi, rolling stats, transformacje GPR, zmienne czasowe."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_COLS = ("gold", "silver")
GPR_COLS = ("gpr", "gpr_act", "gpr_threat")


# --------------------------------------------------------------------------- #
# Podstawowe transformacje
# --------------------------------------------------------------------------- #
def add_returns(df: pd.DataFrame, cols=PRICE_COLS) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[f"{c}_logret"] = np.log(df[c]).diff()
            df[f"{c}_ret"] = df[c].pct_change()
    return df


def add_log_prices(df: pd.DataFrame, cols=PRICE_COLS) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[f"{c}_log"] = np.log(df[c])
    return df


# --------------------------------------------------------------------------- #
# Lagi i statystyki kroczące
# --------------------------------------------------------------------------- #
def add_lags(df: pd.DataFrame, cols, lags) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        for lag in lags:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)
    return df


def add_rolling(df: pd.DataFrame, cols, windows) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        for w in windows:
            df[f"{c}_mean{w}"] = s.rolling(w).mean()
            df[f"{c}_std{w}"] = s.rolling(w).std()
            df[f"{c}_min{w}"] = s.rolling(w).min()
            df[f"{c}_max{w}"] = s.rolling(w).max()
    return df


# --------------------------------------------------------------------------- #
# Specyficzne dla GPR: szoki, z-score, momentum
# --------------------------------------------------------------------------- #
def add_gpr_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "gpr" not in df.columns:
        return df

    df["gpr_chg_1d"] = df["gpr"].diff()
    df["gpr_chg_5d"] = df["gpr"].diff(5)
    df["gpr_chg_20d"] = df["gpr"].diff(20)

    # z-score względem 1-rocznego okna — wykrywa geopolityczne „szoki”
    ma = df["gpr"].rolling(252).mean()
    sd = df["gpr"].rolling(252).std()
    df["gpr_z252"] = (df["gpr"] - ma) / sd

    # flaga ekstremalnego ryzyka
    df["gpr_shock"] = (df["gpr_z252"] > 2).astype(int)

    if "gpr_act" in df.columns and "gpr_threat" in df.columns:
        df["gpr_threat_minus_act"] = df["gpr_threat"] - df["gpr_act"]
    return df


# --------------------------------------------------------------------------- #
# Cechy czasowe
# --------------------------------------------------------------------------- #
def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = pd.to_datetime(df["date"])
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["year"] = d.dt.year
    df["dow"] = d.dt.dayofweek
    # sinusoidalne kodowanie sezonowości
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


# --------------------------------------------------------------------------- #
# Budowa targetu
# --------------------------------------------------------------------------- #
def make_target(df: pd.DataFrame, asset: str, horizon: int, mode: str) -> pd.DataFrame:
    """`mode='return'` → log-zwrot na `horizon` dni w przód, `'price'` → cena."""
    df = df.copy()
    if asset not in df.columns:
        raise ValueError(f"Brak kolumny {asset}")
    if mode == "return":
        df["target"] = np.log(df[asset].shift(-horizon) / df[asset])
    elif mode == "price":
        df["target"] = df[asset].shift(-horizon)
    else:
        raise ValueError(f"Nieznany mode: {mode}")
    return df


# --------------------------------------------------------------------------- #
# Pełny pipeline
# --------------------------------------------------------------------------- #
def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fcfg = cfg["features"]
    tcfg = cfg["target"]

    out = df.copy()
    if fcfg.get("use_returns", True):
        out = add_returns(out)
    if fcfg.get("use_log_prices", True):
        out = add_log_prices(out)

    base_cols = [c for c in (*PRICE_COLS, *GPR_COLS, "gold_silver_ratio") if c in out.columns]
    out = add_lags(out, base_cols, fcfg["lags"])
    out = add_rolling(out, base_cols, fcfg["rolling_windows"])

    out = add_gpr_features(out)
    out = add_calendar(out)

    out = make_target(out, tcfg["asset"], tcfg["horizon_days"], tcfg["mode"])
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Lista kolumn używanych jako X (wszystko poza date, target i surowymi cenami)."""
    drop = {"date", "target", "gold", "silver"}
    return [c for c in df.columns if c not in drop and df[c].dtype != "O"]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.dropna(subset=["target"]).copy()
    cols = feature_columns(df)
    X = df[cols].copy()
    y = df["target"].copy()
    # usuwamy wiersze z NaN w features (efekt lagów/rollingów)
    mask = ~X.isna().any(axis=1)
    return X.loc[mask], y.loc[mask]
