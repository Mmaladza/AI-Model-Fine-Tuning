# Gold & Silver Price Forecasting under Geopolitical Risk

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![CI](https://github.com/Mmaladza/gold-silver-gpr-forecast/actions/workflows/ci.yml/badge.svg)](./.github/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> 🇵🇱 Polska wersja: [README.pl.md](README.pl.md)

End-to-end machine-learning pipeline that forecasts **gold** and **silver** returns
from **daily Geopolitical Risk Index** (GPRD — Caldara & Iacoviello, 2022).
Two hyper-parameter-tuned models (**XGBoost** and a **PyTorch LSTM**) are benchmarked
on 40 years of daily data (1985–2025) and stress-tested against synthetic
geopolitical-shock scenarios.

> Research question: **How much of precious-metal price action can be explained by
> geopolitical risk, and can a model trained on 40 years of data generate actionable
> signals when the next crisis hits?**

---

## Highlights

- **Two fine-tuned models** on the same feature set for fair comparison:
  - gradient-boosted trees (XGBoost) tuned by **Optuna + TPE** over 8 hyper-parameters,
  - a sequence model (LSTM) tuned over architecture (`hidden`, `layers`, `dropout`)
    and training (`lr`, `seq_len`, `batch_size`) with early-stopping.
- **Time-aware validation** — 5-fold `TimeSeriesSplit` for XGB, chronological hold-out
  for LSTM. No look-ahead leakage.
- **Domain-specific features** — not just generic lags: z-scored GPR regime,
  shock flag (>2σ), `threat − act` spread, seasonal sin/cos.
- **Scenario analysis** — multiplies the last 60 days of GPR by
  `{0.5×, 1×, 1.5×, 2×, 3×}` and reports model-implied price response; a sensitivity
  check directly relevant to *safe-haven* strategies.
- **Reproducibility** — one notebook runs the full pipeline end-to-end, including
  Kaggle download and config update. CI tests the core transforms on synthetic data.

---

## Tech stack

| Layer | Tools |
|---|---|
| Data / features | `pandas`, `numpy` |
| Classical ML | `xgboost`, `scikit-learn` |
| Deep learning | `torch` (LSTM, SmoothL1, ReduceLROnPlateau, early-stopping) |
| Hyper-parameter tuning | `optuna` (TPESampler) |
| Viz / EDA | `matplotlib`, `seaborn` |
| Ingestion | `kagglehub` |
| Quality | `pytest`, `ruff`, pre-commit, GitHub Actions |

---

## Project structure

```
.
├── analysis.ipynb              # end-to-end notebook (EDA → tuning → scenarios)
├── configs/config.yaml         # paths, hyper-params, target, tuning budget
├── data/                       # raw CSV + processed cache (git-ignored)
├── models/                     # trained artefacts: xgb.joblib, lstm.pt
├── reports/                    # metrics JSON
├── src/
│   ├── data.py                 # loader + column auto-detect + business-day grid
│   ├── features.py             # lags, rolling stats, GPR-specific transforms
│   ├── models.py               # XGBoost wrapper + LSTM (PyTorch)
│   ├── train.py                # Optuna tuning + time-series CV
│   └── predict.py              # inference + scenario analysis
├── tests/                      # pytest smoke tests on synthetic data
├── .github/workflows/ci.yml    # lint + test on push / PR
└── requirements.txt
```

---

## Quickstart

```bash
# 1. environment
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell / cmd
# source .venv/bin/activate     # Linux / macOS
pip install jupyter

# 2. open the notebook — it installs deps and pulls data from Kaggle on first run
jupyter notebook analysis.ipynb
```

The notebook will:
1. install missing packages from `requirements.txt` (it diffs against the
   currently installed versions, so re-runs are instant),
2. download the dataset via `kagglehub` (needs a Kaggle token — see below),
3. update `configs/config.yaml` with the actual CSV filename,
4. run EDA → feature engineering → XGBoost tuning → LSTM tuning → scenario analysis.

### Kaggle token

1. https://www.kaggle.com/settings → *Create New API Token* → downloads `kaggle.json`.
2. Put it at `C:\Users\<user>\.kaggle\kaggle.json` (Windows) or
   `~/.kaggle/kaggle.json` (Linux/macOS). Or set `KAGGLE_USERNAME` /
   `KAGGLE_KEY` env vars (the notebook shows how).

### CLI alternative

```bash
python -m src.train --model xgb     # tune & train XGBoost
python -m src.train --model lstm    # tune & train LSTM
python -m src.train --model both
python -m src.predict               # forecast + geopolitical scenarios
```

---

## Methodology

### Target
Log-return of the chosen asset (gold by default) over a **20-trading-day horizon**
(≈ one calendar month). All hyper-parameters live in
[configs/config.yaml](configs/config.yaml) — switch to silver, change the horizon
or toggle between return / price mode without touching code.

### Feature engineering ([`src/features.py`](src/features.py))
- log-prices and log-returns for gold & silver,
- lags `[1, 2, 3, 5, 10, 20, 60]` on prices, returns and GPR,
- rolling `mean / std / min / max` over `{5, 20, 60}` days,
- **GPR-specific**: 1/5/20-day changes, 252-day z-score (detects regime shifts),
  `gpr_shock` binary flag (>2σ), `GPRD_THREAT − GPRD_ACT` spread,
- calendar: month, quarter, day-of-week, sin/cos monthly seasonality.

### Validation

| Concern | Handling |
|---|---|
| Look-ahead leakage | `TimeSeriesSplit` (XGB), chronological split (LSTM); features all shifted before target computation |
| Regime shift | Train on 1985–2023, evaluate on the last 24 months |
| Overfit to tuning signal | Optuna objective averaged over 5 CV folds for XGB; separate val set for LSTM |

### Metrics (stored in [reports/](reports/))

| Metric | What it measures | Why it matters |
|---|---|---|
| MAE | mean absolute error of the log-return | robust to outliers |
| RMSE | same but quadratic | penalises large misses |
| **hit_rate** | % of correct direction (up / down) | the only metric that matters for a trading rule |

### Scenario analysis ([`src/predict.py`](src/predict.py))
Multiplies the last 60 days of GPRD by `{0.5, 1.0, 1.5, 2.0, 3.0}`, recomputes
every GPR-derived feature, and asks each model for a fresh forecast.
Output: a small table of `scenario × implied return × change %`, plus the top-20
feature importances — a quick sanity check that GPR actually moves the model.

---

## Results

Running the notebook end-to-end on the full dataset produces:

| Model   | MAE (20-day log-ret) | RMSE   | Hit-rate |
|---------|---------------------:|-------:|---------:|
| XGBoost | _filled after training_ | _…_  | _…_      |
| LSTM    | _filled after training_ | _…_  | _…_      |

The `reports/*.json` files are regenerated every run; plug in your numbers after
the first full execution. Optuna studies can also be persisted to SQLite for
dashboard-style inspection.

---

## Roadmap

- [ ] macro context: DXY, 10-year US yields, VIX, CPI surprise
- [ ] multi-horizon head (5 / 20 / 60 days simultaneously)
- [ ] conformal prediction for calibrated intervals
- [ ] SHAP attribution — how much of each forecast comes from GPR vs price history
- [ ] export: lightweight Streamlit demo of the scenario panel

---

## Data & citation

- Dataset: [Gold-Silver Price vs Geopolitical Risk (1985-2025)](https://www.kaggle.com/datasets/shreyanshdangi/gold-silver-price-vs-geopolitical-risk-19852025) — S. Dangi, Kaggle.
- Underlying index: **Caldara, Dario and Matteo Iacoviello (2022).**
  *Measuring Geopolitical Risk.* American Economic Review, 112 (4), 1194–1225.
  [matteoiacoviello.com/gpr.htm](https://www.matteoiacoviello.com/gpr.htm).

See [CITATION.cff](CITATION.cff) for machine-readable credit.

## License

[MIT](LICENSE) — free to use, learn from, and extend.
