"""Testy inżynierii cech."""

from __future__ import annotations

import numpy as np

from src.data import clean, load_raw
from src.features import (
    add_gpr_features,
    add_lags,
    add_returns,
    build_features,
    feature_columns,
    make_target,
    split_xy,
)


def test_returns_are_computed(cfg):
    df = clean(load_raw(cfg))
    out = add_returns(df)
    assert "gold_logret" in out.columns
    assert "silver_ret" in out.columns
    # log-zwroty powinny mieć średnią bliską zeru i skończoną wariancję
    assert np.isfinite(out["gold_logret"].dropna().std())


def test_lags_shift_values(cfg):
    df = clean(load_raw(cfg))
    out = add_lags(df, ["gold"], [1, 5])
    # lag-1 musi być wczorajszą wartością
    assert (out["gold_lag1"].dropna().values == df["gold"].shift(1).dropna().values).all()


def test_gpr_features_include_shock_flag(cfg):
    df = clean(load_raw(cfg))
    out = add_gpr_features(df)
    assert "gpr_z252" in out.columns
    assert "gpr_shock" in out.columns
    assert set(out["gpr_shock"].dropna().unique()).issubset({0, 1})


def test_make_target_shifts_horizon(cfg):
    df = clean(load_raw(cfg))
    out = make_target(df, asset="gold", horizon=5, mode="return")
    # ostatnie 5 wierszy targetu to NaN (nie ma przyszłości)
    assert out["target"].tail(5).isna().all()


def test_full_pipeline_produces_usable_xy(cfg):
    df = clean(load_raw(cfg))
    feats = build_features(df, cfg)
    X, y = split_xy(feats)
    assert len(X) == len(y) > 0
    assert not X.isna().any().any()
    assert y.notna().all()
    # żadna kolumna featurowa nie może być "raw ceną"
    assert "gold" not in feature_columns(feats)
    assert "silver" not in feature_columns(feats)
