# Prognozowanie cen złota i srebra w kontekście ryzyka geopolitycznego

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![CI](https://github.com/Mmaladza/gold-silver-gpr-forecast/actions/workflows/ci.yml/badge.svg)](./.github/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> 🇬🇧 English version: [README.md](README.md)

Kompletny pipeline ML, który prognozuje **log-zwroty cen złota i srebra**
na podstawie **dziennego indeksu ryzyka geopolitycznego** (GPRD — Caldara & Iacoviello, 2022).
Dwa modele po fine-tuningu (**XGBoost** i **LSTM w PyTorch**) porównywane na 40 latach
danych dziennych (1985–2025) i testowane w scenariuszach syntetycznych szoków
geopolitycznych.

> Pytanie badawcze: **Na ile ryzyko geopolityczne wyjaśnia ruchy cen metali
> szlachetnych i czy model wytrenowany na 40 latach danych potrafi wygenerować
> użyteczne sygnały, gdy pojawia się kolejny kryzys?**

---

## Najważniejsze cechy projektu

- **Dwa sfine-tunowane modele** na tym samym zbiorze cech — uczciwe porównanie:
  - gradient-boosted trees (XGBoost) strojony przez **Optuna + TPE** na 8 hiperparametrach,
  - model sekwencyjny (LSTM) strojony pod kątem architektury (`hidden`, `layers`,
    `dropout`) i treningu (`lr`, `seq_len`, `batch_size`) z wczesnym zatrzymaniem.
- **Walidacja świadoma czasu** — 5-krotny `TimeSeriesSplit` dla XGB, chronologiczny
  hold-out dla LSTM. Zero data-leakage.
- **Cechy specyficzne dla domeny** — nie tylko generyczne lagi: z-score GPR (reżim
  geopolityczny), flaga szoku (>2σ), spread `threat − act`, sezonowość sin/cos.
- **Analiza scenariuszy** — mnoży ostatnie 60 dni GPR przez
  `{0.5×, 1×, 1.5×, 2×, 3×}` i raportuje odpowiedź modelu; bezpośrednia ocena
  wrażliwości kluczowa dla strategii typu *safe-haven*.
- **Powtarzalność** — jeden notebook uruchamia całość end-to-end, włącznie
  z pobraniem danych z Kaggle i aktualizacją configu. CI testuje logikę
  transformacji na syntetycznych danych — nie wymaga tokenu Kaggle.

---

## Stack technologiczny

| Warstwa | Narzędzia |
|---|---|
| Dane / cechy | `pandas`, `numpy` |
| Klasyczne ML | `xgboost`, `scikit-learn` |
| Deep learning | `torch` (LSTM, SmoothL1, ReduceLROnPlateau, early-stopping) |
| Tuning hiperparametrów | `optuna` (TPESampler) |
| Wizualizacja / EDA | `matplotlib`, `seaborn` |
| Pobieranie danych | `kagglehub` |
| Jakość kodu | `pytest`, `ruff`, pre-commit, GitHub Actions |

---

## Struktura projektu

```
.
├── analysis.ipynb              # notebook end-to-end (EDA → tuning → scenariusze)
├── configs/config.yaml         # ścieżki, hiperparametry, target, budżet tuningu
├── data/                       # surowy CSV + cache (ignorowane przez git)
├── models/                     # wytrenowane artefakty: xgb.joblib, lstm.pt
├── reports/                    # metryki JSON
├── src/
│   ├── data.py                 # loader + auto-detekcja kolumn + siatka business-day
│   ├── features.py             # lagi, rolling stats, transformacje GPR
│   ├── models.py               # XGBoost wrapper + LSTM (PyTorch)
│   ├── train.py                # tuning Optuna + walidacja time-series
│   └── predict.py              # inferencja + analiza scenariuszy
├── tests/                      # smoke-testy pytest na syntetycznych danych
├── .github/workflows/ci.yml    # lint + testy na push / PR
└── requirements.txt
```

---

## Szybki start

```bash
# 1. środowisko
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell / cmd
# source .venv/bin/activate     # Linux / macOS
pip install jupyter

# 2. otwórz notebook — przy pierwszym uruchomieniu zainstaluje zależności i pobierze dane
jupyter notebook analysis.ipynb
```

Notebook:
1. zainstaluje brakujące paczki z `requirements.txt` (porównuje wersje
   z już zainstalowanymi, więc kolejne uruchomienia są natychmiastowe),
2. pobierze dataset przez `kagglehub` (wymaga tokenu Kaggle — patrz niżej),
3. zaktualizuje `configs/config.yaml` rzeczywistą nazwą pliku CSV,
4. przeprowadzi EDA → inżynierię cech → tuning XGBoost → tuning LSTM → analizę scenariuszy.

### Token Kaggle

1. https://www.kaggle.com/settings → *Create New API Token* → pobierze się `kaggle.json`.
2. Umieść plik w `C:\Users\<user>\.kaggle\kaggle.json` (Windows) lub
   `~/.kaggle/kaggle.json` (Linux/macOS). Alternatywnie ustaw zmienne środowiskowe
   `KAGGLE_USERNAME` / `KAGGLE_KEY` (w notebooku jest komórka pokazująca jak).

### Alternatywa — z linii poleceń

```bash
python -m src.train --model xgb     # tuning + trening XGBoost
python -m src.train --model lstm    # tuning + trening LSTM
python -m src.train --model both
python -m src.predict               # prognoza + scenariusze geopolityczne
```

---

## Metodologia

### Cel predykcji
Log-zwrot wybranego aktywa (domyślnie złoto) na **horyzoncie 20 dni roboczych**
(≈ jeden miesiąc kalendarzowy). Wszystkie hiperparametry są w
[configs/config.yaml](configs/config.yaml) — możesz przełączyć na srebro, zmienić
horyzont lub tryb (`return` vs `price`) bez ruszania kodu.

### Inżynieria cech ([`src/features.py`](src/features.py))
- log-ceny i log-zwroty dla złota i srebra,
- lagi `[1, 2, 3, 5, 10, 20, 60]` na cenach, zwrotach i GPR,
- rolling `mean / std / min / max` w oknach `{5, 20, 60}` dni,
- **specyficzne dla GPR**: zmiany 1d/5d/20d, z-score na 252 dni (wykrywa reżimy),
  binarna flaga `gpr_shock` (>2σ), spread `GPRD_THREAT − GPRD_ACT`,
- kalendarz: miesiąc, kwartał, dzień tygodnia, sezonowość sin/cos.

### Walidacja

| Ryzyko | Sposób mitygacji |
|---|---|
| Look-ahead leakage | `TimeSeriesSplit` (XGB), split chronologiczny (LSTM); wszystkie cechy przesunięte przed policzeniem targetu |
| Zmiana reżimu | Trening 1985–2023, ocena na ostatnich 24 miesiącach |
| Przeuczenie do sygnału tuningu | Cel Optuny uśredniany po 5 foldach CV (XGB); osobny val set dla LSTM |

### Metryki (zapisywane do [reports/](reports/))

| Metryka | Co mierzy | Dlaczego ma znaczenie |
|---|---|---|
| MAE | średni bezwzględny błąd log-zwrotu | odporna na outliery |
| RMSE | pierwiastek błędu kwadratowego | silniej karze duże pomyłki |
| **hit_rate** | % trafnych kierunków (up / down) | jedyna metryka licząca się dla reguły tradingowej |

### Analiza scenariuszy ([`src/predict.py`](src/predict.py))
Mnoży ostatnie 60 dni GPRD przez `{0.5, 1.0, 1.5, 2.0, 3.0}`, przelicza wszystkie
cechy pochodne i pyta model o prognozę. Wynik: mała tabela
`scenariusz × implikowany zwrot × zmiana %` plus ranking top-20 cech — szybki
sanity-check, czy GPR rzeczywiście napędza model.

---

## Wyniki

Uruchomienie notebooka end-to-end produkuje:

| Model   | MAE (20-d log-ret) | RMSE | Hit-rate |
|---------|-------------------:|-----:|---------:|
| XGBoost | _uzupełnić po treningu_ | _…_ | _…_     |
| LSTM    | _uzupełnić po treningu_ | _…_ | _…_     |

Pliki `reports/*.json` są regenerowane przy każdym treningu — wartości można
stamtąd przenieść. Studia Optuny można dodatkowo zapisywać do SQLite pod
dashboard inspekcyjny.

---

## Dalsze kroki

- [ ] kontekst makro: DXY, 10-letnie rentowności USA, VIX, niespodzianki CPI
- [ ] głowa multi-horizon (równolegle 5 / 20 / 60 dni)
- [ ] conformal prediction — skalibrowane przedziały ufności
- [ ] atrybucja SHAP — ile każdej prognozy pochodzi z GPR, a ile z historii ceny
- [ ] eksport: lekkie demo Streamlit panelu scenariuszowego

---

## Dane i cytowania

- Dataset: [Gold-Silver Price vs Geopolitical Risk (1985-2025)](https://www.kaggle.com/datasets/shreyanshdangi/gold-silver-price-vs-geopolitical-risk-19852025) — S. Dangi, Kaggle.
- Podstawowy indeks: **Caldara, Dario and Matteo Iacoviello (2022).**
  *Measuring Geopolitical Risk.* American Economic Review, 112 (4), 1194–1225.
  [matteoiacoviello.com/gpr.htm](https://www.matteoiacoviello.com/gpr.htm).

Maszynowo czytelne źródła w [CITATION.cff](CITATION.cff).

## Licencja

[MIT](LICENSE) — do swobodnego użytku, uczenia się i rozwijania.
