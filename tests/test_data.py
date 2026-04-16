"""Testy loadera i czyszczenia."""

from __future__ import annotations

import pandas as pd

from src.data import clean, detect_columns, load_raw


def test_detect_columns_finds_daily_gpr(synthetic_raw, cfg):
    mapping = detect_columns(synthetic_raw, cfg)
    assert mapping["date"] == "DATE"
    assert mapping["gold"] == "GOLD_PRICE"
    assert mapping["silver"] == "SILVER_PRICE"
    assert mapping["gpr"] == "GPRD"
    assert mapping["gpr_act"] == "GPRD_ACT"
    assert mapping["gpr_threat"] == "GPRD_THREAT"


def test_load_raw_renames_and_parses_dates(cfg):
    df = load_raw(cfg)
    assert {"date", "gold", "silver", "gpr"}.issubset(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["gold"].notna().all()


def test_clean_adds_ratio_and_business_day_grid(cfg):
    df = clean(load_raw(cfg))
    assert "gold_silver_ratio" in df.columns
    # siatka business-day — różnice między sąsiednimi datami są <= 3 dni (weekendy)
    deltas = df["date"].diff().dropna().dt.days
    assert deltas.max() <= 3
    assert df.isna().sum().sum() == 0
