"""Ładowanie i czyszczenie danych Gold / Silver / GPR (Kaggle 1985-2025).

Dataset: https://www.kaggle.com/datasets/shreyanshdangi/gold-silver-price-vs-geopolitical-risk-19852025
Pobierz CSV ręcznie (`kaggle datasets download ...`) i umieść w `data/`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import yaml


# --------------------------------------------------------------------------- #
# Konfiguracja
# --------------------------------------------------------------------------- #
def load_config(path: str | Path = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------- #
# Auto-wykrywanie kolumn
# --------------------------------------------------------------------------- #
def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def detect_columns(df: pd.DataFrame, cfg: dict) -> dict:
    cc = cfg["columns"]
    mapping = {
        "date": _find_column(df, cc["date_candidates"]),
        "gold": _find_column(df, cc["gold_candidates"]),
        "silver": _find_column(df, cc["silver_candidates"]),
        "gpr": _find_column(df, cc["gpr_candidates"]),
        "gpr_act": _find_column(df, cc["gpr_act_candidates"]),
        "gpr_threat": _find_column(df, cc["gpr_threat_candidates"]),
    }
    missing = [k for k in ("date", "gold", "silver", "gpr") if mapping[k] is None]
    if missing:
        raise ValueError(f"Nie znaleziono kolumn: {missing}. Dostępne: {list(df.columns)}")
    return mapping


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #
def load_raw(cfg: dict) -> pd.DataFrame:
    """Wczytuje surowy plik CSV i ujednolica nazwy kolumn."""
    csv_path = Path(cfg["paths"]["raw_csv"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Brak pliku {csv_path}. Pobierz dataset z Kaggle i zapisz tam.")

    df = pd.read_csv(csv_path)
    col = detect_columns(df, cfg)

    rename = {col[k]: k for k in col if col[k] is not None}
    df = df.rename(columns=rename)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ("gold", "silver", "gpr", "gpr_act", "gpr_threat"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = [
        c for c in ("date", "gold", "silver", "gpr", "gpr_act", "gpr_threat") if c in df.columns
    ]
    return df[keep]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Częstotliwość dzienna (robocza) + forward-fill dla luk krótszych niż 5 dni."""
    df = df.set_index("date").sort_index()
    df = df.asfreq("B")  # business-day grid
    df = df.ffill(limit=5).dropna(how="all")
    df = df.dropna(subset=["gold", "silver", "gpr"])
    df["gold_silver_ratio"] = df["gold"] / df["silver"]
    return df.reset_index()


def _cache_path(cfg: dict) -> Path:
    """Ścieżka do cache'u — parquet jeśli skonfigurowany, inaczej pickle."""
    if "processed_parquet" in cfg["paths"]:
        return Path(cfg["paths"]["processed_parquet"])
    return Path(cfg["paths"].get("processed_pickle", "data/processed.pkl"))


def _read_cache(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_pickle(path)


def _write_cache(df: pd.DataFrame, path: Path) -> None:
    """Zapisuje cache; gdy brakuje engine'u parquet, płynnie degraduje do pickle."""
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return
        except ImportError:
            fallback = path.with_suffix(".pkl")
            print(f"[data] pyarrow/fastparquet niedostępne — zapisuję cache jako {fallback}")
            df.to_pickle(fallback)
            return
    df.to_pickle(path)


def load_processed(cfg: dict, force: bool = False) -> pd.DataFrame:
    """Zwraca oczyszczoną ramkę — z cache'em parquet lub pickle."""
    out = _cache_path(cfg)
    pkl_alt = out.with_suffix(".pkl")

    if not force:
        if out.exists():
            return _read_cache(out)
        if pkl_alt.exists():
            return _read_cache(pkl_alt)

    df = clean(load_raw(cfg))
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_cache(df, out)
    return df


# --------------------------------------------------------------------------- #
# Szybka diagnostyka
# --------------------------------------------------------------------------- #
def summary(df: pd.DataFrame) -> pd.DataFrame:
    desc = df.describe(include="all").T
    desc["missing"] = df.isna().sum()
    desc["missing_pct"] = (df.isna().mean() * 100).round(2)
    return desc


if __name__ == "__main__":  # pragma: no cover
    cfg = load_config()
    df = load_processed(cfg, force=True)
    print(df.head())
    print("\nZakres dat:", df["date"].min(), "→", df["date"].max())
    print("Obserwacji:", len(df))
    print("\nPodsumowanie:")
    print(summary(df))
