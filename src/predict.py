"""Wnioskowanie + analiza scenariuszy geopolitycznych.

Pokazuje jak prognozować cenę dla hipotetycznych wartości GPR:
- baseline  : GPR = aktualny
- łagodny   : GPR × 0.5   (rozluźnienie napięć)
- ostry szok: GPR × 2.0   (kryzys, np. wojna, sankcje)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import load_config, load_processed
from src.features import build_features, feature_columns


# --------------------------------------------------------------------------- #
# Ładowanie modelu
# --------------------------------------------------------------------------- #
def load_xgb(cfg: dict):
    path = Path(cfg["paths"]["models_dir"]) / "xgb.joblib"
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# Prognozowanie na najnowszym wierszu
# --------------------------------------------------------------------------- #
def latest_row(cfg: dict) -> pd.DataFrame:
    df = load_processed(cfg)
    feats = build_features(df, cfg)
    feats = feats.dropna(subset=feature_columns(feats))
    return feats.tail(1)


def predict_latest(cfg: dict) -> dict:
    bundle = load_xgb(cfg)
    model = bundle["model"]
    cols = bundle["features"]

    row = latest_row(cfg)
    X = row[cols]
    y_hat = float(model.predict(X)[0])

    mode = cfg["target"]["mode"]
    asset = cfg["target"]["asset"]
    horizon = cfg["target"]["horizon_days"]
    last_price = float(row[asset].iloc[0])

    forecast_price = last_price * np.exp(y_hat) if mode == "return" else y_hat

    return {
        "date": pd.to_datetime(row["date"].iloc[0]).strftime("%Y-%m-%d"),
        "asset": asset,
        "last_price": last_price,
        "horizon_days": horizon,
        "pred_log_return" if mode == "return" else "pred_price": y_hat,
        "forecast_price": forecast_price,
    }


# --------------------------------------------------------------------------- #
# Analiza scenariuszy geopolitycznych
# --------------------------------------------------------------------------- #
def scenario_predictions(
    cfg: dict,
    gpr_multipliers: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Rekalkulacja cech przy zmodyfikowanym GPR i prognoza ceny.

    Parameters
    ----------
    gpr_multipliers : dict
        np. {"spokój": 0.5, "baseline": 1.0, "napięcie": 1.5, "kryzys": 2.0}
    """
    gpr_multipliers = gpr_multipliers or {
        "spokój (GPR×0.5)": 0.5,
        "baseline (GPR×1.0)": 1.0,
        "napięcie (GPR×1.5)": 1.5,
        "kryzys (GPR×2.0)": 2.0,
        "szok ekstremalny (GPR×3.0)": 3.0,
    }

    bundle = load_xgb(cfg)
    model = bundle["model"]
    cols = bundle["features"]

    df = load_processed(cfg)
    asset = cfg["target"]["asset"]
    mode = cfg["target"]["mode"]

    rows = []
    for label, mult in gpr_multipliers.items():
        shocked = df.copy()
        # mnożymy ostatnie 60 obserwacji GPR — tak, by lagi/rollingi
        # zdążyły odzwierciedlić nowy reżim
        tail_idx = shocked.index[-60:]
        for c in ("gpr", "gpr_act", "gpr_threat"):
            if c in shocked.columns:
                shocked.loc[tail_idx, c] = shocked.loc[tail_idx, c] * mult

        feats = build_features(shocked, cfg)
        feats = feats.dropna(subset=feature_columns(feats))
        X = feats.tail(1)[cols]
        y_hat = float(model.predict(X)[0])

        last_price = float(shocked[asset].iloc[-1])
        forecast = last_price * np.exp(y_hat) if mode == "return" else y_hat
        rows.append(
            {
                "scenariusz": label,
                "mnożnik_GPR": mult,
                "GPR_end": float(shocked["gpr"].iloc[-1]),
                f"{asset}_last": last_price,
                "pred_log_return" if mode == "return" else "pred_price": y_hat,
                "forecast_price": forecast,
                "change_pct": (forecast / last_price - 1) * 100,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Ważność cech (wgląd w model)
# --------------------------------------------------------------------------- #
def feature_importances(cfg: dict, top: int = 20) -> pd.DataFrame:
    bundle = load_xgb(cfg)
    model = bundle["model"]
    cols = bundle["features"]
    imp = model.feature_importances_
    return (
        pd.DataFrame({"feature": cols, "importance": imp})
        .sort_values("importance", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    cfg = load_config()
    print("== Prognoza na najnowszych danych ==")
    print(predict_latest(cfg))
    print("\n== Scenariusze geopolityczne ==")
    print(scenario_predictions(cfg).to_string(index=False))
    print("\n== Top 20 cech ==")
    print(feature_importances(cfg).to_string(index=False))


if __name__ == "__main__":
    main()
