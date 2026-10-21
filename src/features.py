"""
src/features.py
---------------
Turns the daily weather table into a supervised learning table.

Framing (important for avoiding leakage):
  Each row is a FORECAST ORIGIN = the last day we have observations for.
  Every feature is computed ONLY from data on or before the origin day.
  The target is the temperature `horizon` days AFTER the origin.

  For next-day forecasting (horizon=1) the feature "temp_mean_lag1" is
  simply today's mean temperature, "temp_mean_lag2" is yesterday's, and so
  on -- i.e. lags of 1, 2, 3, 7 and 14 days relative to the forecast day.
  Rolling statistics also end at the origin day, never after it.
  The only "future" information used is the calendar (day-of-year of the
  target day), which is known in advance and cannot leak weather data.
"""

import numpy as np
import pandas as pd

LAG_DAYS = [1, 2, 3, 7, 14]          # lag of temp_mean, relative to forecast day
ROLL_WINDOWS = [3, 7, 30]            # rolling mean/std windows (days)
RAIN_THRESHOLD_MM = 1.0              # >= 1 mm tomorrow counts as a "rain day"

FEATURE_COLUMNS = (
    [f"temp_mean_lag{k}" for k in LAG_DAYS]
    + ["temp_max_lag1", "temp_min_lag1", "humidity_lag1", "pressure_lag1",
       "wind_speed_max_lag1", "precipitation_lag1", "cloud_cover_lag1",
       "temp_range_lag1", "pressure_change_1d", "pressure_change_3d"]
    + [f"temp_roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"temp_roll_std_{w}" for w in ROLL_WINDOWS]
    + ["doy_sin", "doy_cos"]
)


def _city_features(g, horizon):
    """Build features + targets for one city's daily frame."""
    g = g.sort_values("date").copy()
    t = g["temp_mean"]

    # Lag features: value at (origin - (k-1)) == k days before forecast day.
    for k in LAG_DAYS:
        g[f"temp_mean_lag{k}"] = t.shift(k - 1)

    # Most recent observed values of the other variables (at the origin).
    g["temp_max_lag1"] = g["temp_max"]
    g["temp_min_lag1"] = g["temp_min"]
    g["humidity_lag1"] = g["humidity"]
    g["pressure_lag1"] = g["pressure"]
    g["wind_speed_max_lag1"] = g["wind_speed_max"]
    g["precipitation_lag1"] = g["precipitation"]
    g["cloud_cover_lag1"] = g["cloud_cover"]
    g["temp_range_lag1"] = g["temp_max"] - g["temp_min"]

    # Pressure trend ending at the origin day.
    g["pressure_change_1d"] = g["pressure"] - g["pressure"].shift(1)
    g["pressure_change_3d"] = g["pressure"] - g["pressure"].shift(3)

    # Rolling statistics over windows ENDING at the origin day.
    for w in ROLL_WINDOWS:
        g[f"temp_roll_mean_{w}"] = t.rolling(w).mean()
        g[f"temp_roll_std_{w}"] = t.rolling(w).std()

    # Seasonal features of the TARGET day (calendar only -- no leakage).
    target_date = g["date"] + pd.Timedelta(days=horizon)
    doy = target_date.dt.dayofyear
    g["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    g["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # Targets.
    g["target_temp"] = t.shift(-horizon)
    g["target_rain"] = (g["precipitation"].shift(-1) >= RAIN_THRESHOLD_MM).astype(float)
    g.loc[g["precipitation"].shift(-1).isna(), "target_rain"] = np.nan
    g["target_date"] = target_date
    return g


def build_supervised(daily, horizon=1):
    """Return a supervised DataFrame for the given forecast horizon.

    Columns: date (origin), target_date, city, FEATURE_COLUMNS,
             target_temp, target_rain (rain target only meaningful at h=1).
    Rows with any missing feature or target are dropped.
    """
    frames = [_city_features(g, horizon) for _, g in daily.groupby("city", sort=False)]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_temp"]).reset_index(drop=True)
    return df


def split_time_based(df, test_days=365):
    """Latest `test_days` days (by target date) become the test set."""
    cutoff = df["target_date"].max() - pd.Timedelta(days=test_days)
    train = df[df["target_date"] <= cutoff].copy()
    test = df[df["target_date"] > cutoff].copy()
    return train, test, cutoff


def climatology_table(train_daily):
    """Mean temperature per city + day-of-year, from TRAIN data only.

    Used by the seasonal-average baseline. Computing it on train only is
    essential -- using the full dataset here would leak test information.
    """
    d = train_daily.copy()
    d["doy"] = d["date"].dt.dayofyear
    table = d.groupby(["city", "doy"])["temp_mean"].mean().reset_index()
    return table


def seasonal_baseline_predict(table, cities, target_dates):
    """Predict temperature from the climatology table."""
    td = pd.DataFrame({"city": cities, "target_date": pd.to_datetime(target_dates)})
    td["doy"] = td["target_date"].dt.dayofyear
    merged = td.merge(table, on=["city", "doy"], how="left")
    # Day 366 (leap years) may be missing -> fall back to day 365 values.
    missing = merged["temp_mean"].isna()
    if missing.any():
        fallback = table[table["doy"] == 365].set_index("city")["temp_mean"]
        merged.loc[missing, "temp_mean"] = merged.loc[missing, "city"].map(fallback)
    return merged["temp_mean"].to_numpy()
