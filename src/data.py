"""
src/data.py
-----------
Downloads historical weather data from the Open-Meteo Archive API
(free, no API key) and caches it locally as CSV files, so the project
also runs offline after the first download.

Source variables are HOURLY; they are aggregated to one row per day:
  temp_max / temp_min / temp_mean  - daily max / min / mean temperature (deg C)
  humidity                         - daily mean relative humidity (%)
  pressure                         - daily mean sea-level pressure (hPa)
  wind_speed_max / wind_speed_mean - daily max / mean wind speed (km/h)
  precipitation                    - daily total precipitation (mm)
  cloud_cover                      - daily mean cloud cover (%)
"""

import os
import time
from datetime import date, timedelta

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "weather_daily.csv")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CITIES = {
    "Indore":    {"lat": 22.7196, "lon": 75.8577},
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639},
}

START_DATE = "2014-01-01"
HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "pressure_msl",
    "cloud_cover", "wind_speed_10m", "precipitation",
]


def _fetch_hourly(lat, lon, start, end, retries=3):
    """Fetch hourly data for one date range, with simple retries."""
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Asia/Kolkata",
    }
    for attempt in range(retries):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=120)
            r.raise_for_status()
            return r.json()["hourly"]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def _hourly_to_daily(hourly):
    """Aggregate an Open-Meteo hourly block to daily rows."""
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.floor("D")
    daily = df.groupby("date").agg(
        temp_max=("temperature_2m", "max"),
        temp_min=("temperature_2m", "min"),
        temp_mean=("temperature_2m", "mean"),
        humidity=("relative_humidity_2m", "mean"),
        pressure=("pressure_msl", "mean"),
        wind_speed_max=("wind_speed_10m", "max"),
        wind_speed_mean=("wind_speed_10m", "mean"),
        precipitation=("precipitation", "sum"),
        cloud_cover=("cloud_cover", "mean"),
    ).reset_index()
    return daily


def download_city(city, start=START_DATE, end=None):
    """Download (or load from cache) daily data for one city."""
    os.makedirs(RAW_DIR, exist_ok=True)
    cache = os.path.join(RAW_DIR, f"{city.lower()}.csv")
    if end is None:
        end = (date.today() - timedelta(days=1)).isoformat()

    if os.path.exists(cache):
        cached = pd.read_csv(cache, parse_dates=["date"])
        if cached["date"].max() >= pd.Timestamp(end) - pd.Timedelta(days=2):
            return cached

    print(f"Downloading {city} ({start} to {end}) ...")
    frames = []
    year = int(start[:4])
    while year <= int(end[:4]):
        y_start = max(start, f"{year}-01-01")
        y_end = min(end, f"{year}-12-31")
        hourly = _fetch_hourly(CITIES[city]["lat"], CITIES[city]["lon"], y_start, y_end)
        frames.append(_hourly_to_daily(hourly))
        year += 1
    daily = pd.concat(frames, ignore_index=True)
    daily.insert(1, "city", city)
    daily.to_csv(cache, index=False)
    print(f"  saved {len(daily)} days -> {cache}")
    return daily


def download_all():
    """Download all cities and write the combined processed CSV."""
    frames = [download_city(c) for c in CITIES]
    combined = pd.concat(frames, ignore_index=True).sort_values(["city", "date"])
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    combined.to_csv(PROCESSED_PATH, index=False)
    print(f"Combined dataset: {len(combined)} rows -> {PROCESSED_PATH}")
    return combined


def load_daily():
    """Load the combined daily dataset (downloading first if needed)."""
    if os.path.exists(PROCESSED_PATH):
        return pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    return download_all()


def clean_daily(df):
    """Handle missing values and impossible outliers, per city.

    - Short gaps (<= 3 days) in the reanalysis data are linearly
      interpolated; remaining edge gaps are forward/backward filled.
    - Physically impossible values (e.g. humidity outside 0-100,
      negative precipitation) are treated as missing first.
    """
    df = df.copy()
    df.loc[~df["humidity"].between(0, 100), "humidity"] = pd.NA
    df.loc[~df["cloud_cover"].between(0, 100), "cloud_cover"] = pd.NA
    df.loc[df["precipitation"] < 0, "precipitation"] = pd.NA
    df.loc[~df["pressure"].between(950, 1060), "pressure"] = pd.NA

    numeric = ["temp_max", "temp_min", "temp_mean", "humidity", "pressure",
               "wind_speed_max", "wind_speed_mean", "precipitation", "cloud_cover"]
    out = []
    for _, g in df.groupby("city", sort=False):
        g = g.sort_values("date").set_index("date")
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full_idx)
        g[numeric] = g[numeric].interpolate(method="linear", limit=3)
        g[numeric] = g[numeric].ffill().bfill()
        g["city"] = g["city"].ffill().bfill()
        out.append(g.reset_index().rename(columns={"index": "date"}))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    data = clean_daily(download_all())
    data.to_csv(PROCESSED_PATH, index=False)
    print(data.groupby("city")["date"].agg(["min", "max", "count"]))
