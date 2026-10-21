# Weather Forecasting using Machine Learning

A complete university ML lab project: forecast **tomorrow's temperature**,
the **next 7 days**, and **whether it will rain tomorrow** for 5 Indian
cities (Indore, Delhi, Mumbai, Bengaluru, Kolkata), using 12+ years of real
historical weather from the free [Open-Meteo](https://open-meteo.com)
archive API — with rigorous, leakage-free methodology and a live Streamlit
demo app.

## Quick start (Windows)

Open **Command Prompt** or **PowerShell** in the project folder and run:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

Then open http://localhost:8501 in your browser. The trained models are
already saved in `models/`, so the app starts instantly — no training
needed. (Tested with Python 3.12; other 3.10+ versions should work.)

To retrain everything from scratch (about 15 minutes):

```bat
python train.py
```

To explore the story as a notebook, open
`notebooks/weather_forecasting.ipynb` in VS Code or Jupyter
(`pip install jupyter` if needed).

## Headline results (test year: Oct 2025 – Oct 2026, never used in training)

Next-day mean temperature, all 5 cities (1,825 forecasts):

| Model | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|
| MLP | 0.657 | 0.903 | 0.960 |
| XGBoost | 0.659 | 0.897 | 0.961 |
| **Ridge (winner, by cross-validation)** | **0.672** | **0.923** | **0.959** |
| Random Forest | 0.678 | 0.925 | 0.958 |
| Baseline: Persistence (tomorrow = today) | 0.702 | 0.985 | 0.953 |
| Baseline: Seasonal average | 1.464 | 1.930 | 0.819 |

- **Winner selection is honest:** Ridge was picked by TimeSeriesSplit
  cross-validation on the training years (CV MAE 0.626 °C) *before* touching
  the test year — the test table above was computed afterwards for everyone.
- **7-day outlook:** one direct model per horizon; MAE grows from 0.67 °C
  (day 1) to 1.54 °C (day 7), better than both baselines at every horizon.
- **Prediction intervals:** 80% interval ≈ **± 1.0 °C**, calibrated on a
  held-out slice of the training data (residual quantiles).
- **Rain tomorrow (Random Forest):** F1 **0.812**, ROC-AUC **0.943**,
  precision 0.85, recall 0.78. Rain day = ≥ 1 mm precipitation.

## Project structure

```
data/raw/           Cached per-city downloads (CSV) — project runs offline
data/processed/     Combined daily dataset (23,380 rows)
notebooks/          weather_forecasting.ipynb — the full story, explained
src/data.py         Open-Meteo download, hourly->daily aggregation, cleaning
src/features.py     Leakage-safe feature engineering + time-based split
src/models.py       Model zoo + TimeSeriesSplit tuning helpers
src/evaluate.py     Metrics + all report figures
app/app.py          Streamlit app (Forecast / Model Insights / Data Explorer)
models/             Saved trained models + metadata.json
reports/            metrics.json + figures/ (charts used by the app)
train.py            One command that retrains EVERYTHING
```

## Methodology in one minute

- **Framing:** each row is a *forecast origin* (end of a day). Features use
  only data available at that moment; the target is a future day. Rolling
  windows end at the origin, scalers live inside sklearn Pipelines, and the
  climatology baseline uses training years only — no leakage anywhere.
- **Features:** temperature lags (1/2/3/7/14 days), rolling mean/std
  (3/7/30 days), latest humidity/pressure/wind/precipitation/cloud cover,
  temperature range, pressure change (1d, 3d), sin/cos of day-of-year, city
  (one-hot).
- **Split:** latest 365 days = test; everything earlier = train. Tuning via
  3-fold TimeSeriesSplit — never random CV on time series.
- **Missing values/outliers:** physically impossible values (humidity
  outside 0–100 etc.) are set to missing, short gaps (≤ 3 days) linearly
  interpolated per city.
- **Error analysis:** the model is weakest in June/monsoon (MAE by month in
  `reports/figures/mae_by_month.png`) — monsoon temperatures depend on rain
  events that yesterday's readings can't fully reveal — and strongest in
  calm November weather.

## Design Decisions

- **Shared model + city one-hot** instead of 5 separate models — 5× more
  training data per model, shared patterns (e.g. monsoon) are learned once,
  and there is one pipeline to maintain.
- **Hourly API data aggregated to daily** — guarantees humidity, pressure
  and cloud-cover daily means exist and are computed identically for
  training and for live app forecasts.
- **Direct per-horizon models** for the 7-day forecast instead of recursive
  forecasting — recursive feeding compounds errors day after day.
- **Winner chosen by CV, not by test score** — picking by test score would
  be a subtle form of leakage; the test set is a final exam, not a tutor.
- **MLP included, LSTM not** — sklearn's MLP runs reliably on CPU in
  seconds; a TensorFlow/Keras LSTM adds heavy dependencies and minutes of
  training for no expected gain on daily tabular features.
- **Ridge won, and we kept it** — with strong lag/rolling features, next-day
  temperature is nearly linear. Reporting that honestly beats forcing a
  complex model to "win".
- **Residual-quantile intervals** instead of assuming Gaussian errors —
  simple, distribution-free, and calibrated on data the model never
  trained on.
- **Class weights for rain** (balanced RF / scale_pos_weight XGBoost) —
  rain days are a minority, so unweighted models learn to almost always
  say "no rain" and look deceptively accurate.
- **Live API with cache fallback in the app** — the demo fetches today's
  real weather when online and still works fully offline from the CSV.
- **SHAP via LinearExplainer for the Ridge winner** (TreeExplainer is used
  automatically if a tree model wins a future retrain) — exact SHAP values
  for linear models, no sampling approximation.

## Screenshots

Run the app (see Quick start) to see it live — the Forecast tab shows
tomorrow's temperature with its interval, rain probability, and the 7-day
chart against Open-Meteo's official forecast. All model figures the app
displays — predicted vs actual, error by horizon, feature importance, SHAP
summary, confusion matrix, MAE by month — are in `reports/figures/`.
