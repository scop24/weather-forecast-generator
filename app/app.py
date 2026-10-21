"""
app/app.py
----------
Weather Forecast — Streamlit demo app (premium UI).

Tabs:
  Forecast       -- pick a city, get tomorrow's temperature (+ 80% range),
                    rain probability, and a 7-day forecast comparing our ML
                    model against Open-Meteo's official forecast.
  Model Insights -- leaderboard, error analysis, feature importance, SHAP,
                    confusion matrix (everything produced by train.py).
  Data Explorer  -- interactive charts over the full historical dataset.

Live data: the app first tries to fetch the latest weather from the
Open-Meteo API; if that fails (no internet), it falls back to the cached
CSV in data/processed/ so the demo always works.

Run from the project root:
    streamlit run app/app.py
"""

import json
import os
import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data import CITIES, FORECAST_URL, _hourly_to_daily, load_daily  # noqa: E402
from src.features import FEATURE_COLUMNS, _city_features  # noqa: E402

st.set_page_config(page_title="Weather Forecast", page_icon="🌦️",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Premium theme (custom CSS)
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #0b1226 0%, #10224a 45%, #0e3a5d 100%);
    color: #e8eefc;
}

/* animated glow orbs */
.stApp::before {
    content: ""; position: fixed; top: -180px; right: -120px;
    width: 480px; height: 480px; border-radius: 50%;
    background: radial-gradient(circle, rgba(56,189,248,.35), transparent 65%);
    pointer-events: none; z-index: 0;
}
.stApp::after {
    content: ""; position: fixed; bottom: -220px; left: -140px;
    width: 520px; height: 520px; border-radius: 50%;
    background: radial-gradient(circle, rgba(168,85,247,.28), transparent 65%);
    pointer-events: none; z-index: 0;
}

section[data-testid="stSidebar"] {
    background: rgba(10, 17, 40, 0.85);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * { color: #dbe7ff !important; }

h1, h2, h3 { color: #f2f7ff !important; font-weight: 700 !important; letter-spacing: .2px; }
p, span, label, .stCaption { color: #c7d5f5; }

/* hero card */
.hero {
    background: linear-gradient(120deg, rgba(37,99,235,.85), rgba(14,165,233,.75) 55%, rgba(168,85,247,.65));
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 26px; padding: 30px 36px; margin: 6px 0 20px 0;
    box-shadow: 0 18px 50px rgba(2, 8, 30, .55);
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;
}
.hero .city { font-size: 2.3rem; font-weight: 800; color: #fff; line-height: 1.05; }
.hero .sub  { font-size: 1.0rem; color: rgba(255,255,255,.85); margin-top: 6px; }
.hero .bignum { font-size: 5.2rem; font-weight: 800; color: #fff; line-height: 1; text-align: right; }
.hero .bigsub { font-size: 1.0rem; color: rgba(255,255,255,.9); text-align: right; margin-top: 6px; }
.chip {
    display: inline-block; padding: 5px 14px; margin: 8px 6px 0 0; border-radius: 999px;
    background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
    color: #fff; font-size: .9rem; font-weight: 600;
}

/* glass KPI cards */
.kpi {
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.14);
    border-radius: 18px; padding: 18px 20px; height: 100%;
    box-shadow: 0 8px 28px rgba(2,8,30,.35); backdrop-filter: blur(8px);
}
.kpi .label { font-size: .85rem; text-transform: uppercase; letter-spacing: 1.2px; color: #9db8e8; font-weight: 600; }
.kpi .value { font-size: 2.1rem; font-weight: 800; color: #ffffff; margin: 4px 0 2px 0; }
.kpi .note  { font-size: .92rem; color: #b9cbf2; }

/* 7-day cards */
.days { display: flex; gap: 12px; flex-wrap: wrap; margin: 6px 0 4px 0; }
.day {
    flex: 1 1 120px; text-align: center; padding: 16px 8px 14px 8px; border-radius: 18px;
    background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 8px 24px rgba(2,8,30,.30);
}
.day.today { background: linear-gradient(160deg, rgba(37,99,235,.55), rgba(14,165,233,.35)); border-color: rgba(125,211,252,.6); }
.day .dow { font-weight: 700; color: #fff; font-size: 1.02rem; }
.day .dmy { font-size: .8rem; color: #a9c0ec; }
.day .icon { font-size: 1.9rem; margin: 6px 0 2px 0; }
.day .t { font-size: 1.45rem; font-weight: 800; color: #fff; }
.day .rangebar { height: 6px; border-radius: 99px; margin: 9px 10px 6px 10px;
    background: linear-gradient(90deg, #38bdf8, #fbbf24, #f97316); }
.day .rng { font-size: .8rem; color: #b9cbf2; }

/* tabs */
button[data-baseweb="tab"] {
    font-size: 1.05rem; font-weight: 600; color: #a9c0ec;
    padding: 10px 18px; border-radius: 12px 12px 0 0;
}
button[data-baseweb="tab"][aria-selected="true"] { color: #ffffff; background: rgba(255,255,255,0.08); }

/* buttons + inputs */
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(120deg, #2563eb, #0ea5e9); color: #fff; border: none;
    border-radius: 12px; font-weight: 600; padding: .55rem 1.2rem;
    box-shadow: 0 6px 18px rgba(14,165,233,.35);
}
div[data-baseweb="select"] > div { background: rgba(255,255,255,0.08); border-radius: 12px; border-color: rgba(255,255,255,0.2); }

.section-title { font-size: 1.35rem; font-weight: 700; color: #fff; margin: 14px 0 6px 0; }
.footer { text-align: center; color: #7e97c9; font-size: .85rem; margin: 30px 0 6px 0; }
::-webkit-scrollbar { width: 10px; } ::-webkit-scrollbar-thumb { background: #2b4a7f; border-radius: 8px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def style_fig(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.03)",
        font=dict(color="#dbe7ff", family="Outfit"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=30, b=10, l=10, r=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.10)"),
    )
    if height:
        fig.update_layout(height=height)
    return fig


def condition_icon(temp, rain_prob):
    if rain_prob >= 0.6:
        return "🌧️", "Rain likely"
    if rain_prob >= 0.35:
        return "🌦️", "Rain possible"
    if temp >= 35:
        return "☀️", "Hot & sunny"
    if temp >= 27:
        return "🌤️", "Warm & clear"
    if temp >= 18:
        return "⛅", "Mild & partly cloudy"
    return "🌥️", "Cool & cloudy"


def kpi_card(col, label, value, note):
    col.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="note">{note}</div></div>',
        unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Load saved artifacts (cached so reruns are instant)
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(os.path.join(ROOT, "models", "metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(ROOT, "reports", "metrics.json")) as f:
        metrics = json.load(f)
    horizon_models = joblib.load(os.path.join(ROOT, "models", "horizon_models.joblib"))
    rain_model = joblib.load(os.path.join(ROOT, "models", "rain_classifier.joblib"))
    return metadata, metrics, horizon_models, rain_model


@st.cache_data(ttl=3600)
def load_history():
    return load_daily()


try:
    METADATA, METRICS, HORIZON_MODELS, RAIN_MODEL = load_artifacts()
except Exception:
    st.error("Trained models not found. Please run `python train.py` first, "
             "then reload this page.")
    st.stop()

HISTORY = load_history()
X_COLS = METADATA["feature_columns"]


# --------------------------------------------------------------------------
# Live data helpers
# --------------------------------------------------------------------------
def fetch_live_recent(city, past_days=45):
    """Latest daily weather for a city from the Open-Meteo forecast API,
    aggregated exactly like the training data. Returns None on failure."""
    try:
        params = {
            "latitude": CITIES[city]["lat"], "longitude": CITIES[city]["lon"],
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,"
                      "cloud_cover,wind_speed_10m,precipitation",
            "past_days": past_days, "forecast_days": 1,
            "timezone": "Asia/Kolkata",
        }
        r = requests.get(FORECAST_URL, params=params, timeout=30)
        r.raise_for_status()
        daily = _hourly_to_daily(r.json()["hourly"])
        daily.insert(1, "city", city)
        today = pd.Timestamp(date.today())
        return daily[daily["date"] < today].reset_index(drop=True)
    except Exception:
        return None


def fetch_official_forecast(city, days=7):
    """Open-Meteo's own official daily forecast, for comparison."""
    try:
        params = {
            "latitude": CITIES[city]["lat"], "longitude": CITIES[city]["lon"],
            "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min,"
                     "precipitation_probability_mean",
            "forecast_days": days + 1, "timezone": "Asia/Kolkata",
        }
        r = requests.get(FORECAST_URL, params=params, timeout=30)
        r.raise_for_status()
        d = r.json()["daily"]
        df = pd.DataFrame({
            "date": pd.to_datetime(d["time"]),
            "official_temp_mean": d["temperature_2m_mean"],
            "official_rain_prob": d["precipitation_probability_mean"],
        })
        return df[df["date"] > pd.Timestamp(date.today()) - pd.Timedelta(days=1)]
    except Exception:
        return None


def city_history_with_live(city):
    """Cached history for a city, extended with live data when available."""
    hist = HISTORY[HISTORY["city"] == city].sort_values("date").copy()
    live = fetch_live_recent(city)
    if live is None or live.empty:
        return hist, False
    newer = live[live["date"] > hist["date"].max()]
    if newer.empty:
        return hist, True
    return pd.concat([hist, newer], ignore_index=True), True


def latest_feature_row(city_hist, horizon):
    """Feature vector for the most recent observed day (the forecast origin)."""
    recent = city_hist.tail(120).copy()
    feats = _city_features(recent, horizon=horizon)
    row = feats.iloc[-1]
    X = pd.DataFrame([{c: row[c] for c in FEATURE_COLUMNS}])
    X["city"] = city_hist["city"].iloc[-1]
    return X[X_COLS], row["date"]


def forecast_city(city):
    """Return (results_df, rain_prob, used_live, origin_date). One row per horizon 1..7."""
    hist, used_live = city_history_with_live(city)
    origin_date = hist["date"].max()
    rows = []
    for h in range(1, 8):
        X, _ = latest_feature_row(hist, horizon=h)
        pred = float(HORIZON_MODELS[h].predict(X)[0])
        lo_off, hi_off = METADATA["horizon_intervals_80"][str(h)]
        rows.append({
            "date": origin_date + pd.Timedelta(days=h),
            "horizon": h, "predicted_temp": round(pred, 1),
            "lower": round(pred + lo_off, 1), "upper": round(pred + hi_off, 1),
        })
    X1, _ = latest_feature_row(hist, horizon=1)
    rain_prob = float(RAIN_MODEL.predict_proba(X1)[0][1])
    return pd.DataFrame(rows), rain_prob, used_live, origin_date


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌦️ Weather Forecast")
    st.markdown("Machine-learning weather prediction for 5 Indian cities, "
                "trained on 12+ years of Open-Meteo data.")
    city = st.selectbox("📍 Choose a city", METADATA["cities"],
                        index=METADATA["cities"].index("Indore"))
    st.markdown("---")
    _winner_mae = next((r["MAE"] for r in METRICS["leaderboard"]
                        if r["model"] == METRICS["winner"]), "—")
    st.markdown(f"**Best model:** {METADATA['winner']}  \n"
                f"**Rain model:** {METADATA['rain_winner']}  \n"
                f"**Test MAE (winner):** {_winner_mae} °C  \n"
                f"**Data:** {METADATA['data_range'][0]} → {METADATA['data_range'][1]}")
    st.markdown("---")
    st.caption("Built with scikit-learn + Streamlit · Data: Open-Meteo API")

with st.spinner(f"Preparing forecast for {city} ..."):
    fc, rain_prob, used_live, origin = forecast_city(city)
official = fetch_official_forecast(city)
tomorrow = fc.iloc[0]
off_tomorrow = None
if official is not None and not official.empty:
    match = official[official["date"] == tomorrow["date"]]
    if not match.empty:
        off_tomorrow = float(match["official_temp_mean"].iloc[0])

icon, condition = condition_icon(tomorrow["predicted_temp"], rain_prob)
today_str = pd.Timestamp(date.today()).strftime("%A, %d %B %Y")
src_badge = "🟢 Live · Open-Meteo API" if used_live else "🟠 Cached data (offline mode)"

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
      <div>
        <div class="city">{icon} {city}</div>
        <div class="sub">{today_str} · Tomorrow: {condition}</div>
        <div>
          <span class="chip">🌧️ Rain chance {rain_prob * 100:.0f}%</span>
          <span class="chip">🎯 Range {tomorrow['lower']:.1f} – {tomorrow['upper']:.1f} °C</span>
          <span class="chip">{src_badge}</span>
        </div>
      </div>
      <div>
        <div class="bignum">{tomorrow['predicted_temp']:.0f}°</div>
        <div class="bigsub">predicted mean temperature tomorrow<br/>
        (latest observed: {origin.date()})</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True)

tab_forecast, tab_insights, tab_data = st.tabs(
    ["🔮 Forecast", "📊 Model Insights", "🗂️ Data Explorer"])

# ============================================================== FORECAST ===
with tab_forecast:
    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, "🌡️ Tomorrow (ML model)", f"{tomorrow['predicted_temp']:.1f} °C",
             f"80% range {tomorrow['lower']:.1f} – {tomorrow['upper']:.1f} °C")
    kpi_card(c2, "🌧️ Rain probability", f"{rain_prob * 100:.0f} %",
             condition + " tomorrow")
    if off_tomorrow is not None:
        diff = tomorrow["predicted_temp"] - off_tomorrow
        kpi_card(c3, "🛰️ Open-Meteo official", f"{off_tomorrow:.1f} °C",
                 f"Our model is {diff:+.1f} °C vs official")
    else:
        kpi_card(c3, "🛰️ Open-Meteo official", "—", "Unavailable (offline)")
    kpi_card(c4, "📅 7-day outlook",
             f"{fc['predicted_temp'].min():.0f}° – {fc['predicted_temp'].max():.0f} °C",
             "Min – max across the week")

    st.markdown('<div class="section-title">🗓️ Your week ahead</div>',
                unsafe_allow_html=True)
    week_min, week_max = fc["lower"].min(), fc["upper"].max()
    cards = ['<div class="days">']
    for i, (_, r) in enumerate(fc.iterrows()):
        ic, _ = condition_icon(r["predicted_temp"], rain_prob if i == 0 else 0.15)
        left = 100 * (r["lower"] - week_min) / max(week_max - week_min, 0.1)
        width = max(100 * (r["upper"] - r["lower"]) / max(week_max - week_min, 0.1), 8)
        cards.append(
            f'<div class="day {"today" if i == 0 else ""}">'
            f'<div class="dow">{"Tomorrow" if i == 0 else r["date"].strftime("%a")}</div>'
            f'<div class="dmy">{r["date"].strftime("%d %b")}</div>'
            f'<div class="icon">{ic}</div>'
            f'<div class="t">{r["predicted_temp"]:.0f}°</div>'
            f'<div class="rangebar" style="margin-left:{left:.0f}%; width:{width:.0f}%;"></div>'
            f'<div class="rng">{r["lower"]:.0f}° – {r["upper"]:.0f}°</div></div>')
    cards.append("</div>")
    st.markdown("".join(cards), unsafe_allow_html=True)

    st.markdown('<div class="section-title">📈 7-day forecast — ML model vs official</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["upper"], mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["lower"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(56,189,248,0.22)",
        name="80% interval"))
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["predicted_temp"], mode="lines+markers",
        line=dict(color="#38bdf8", width=3.5, shape="spline"),
        marker=dict(size=9, color="#fff", line=dict(color="#38bdf8", width=2)),
        name="Our ML model"))
    if official is not None and not official.empty:
        merged = fc.merge(official, on="date", how="inner")
        if not merged.empty:
            fig.add_trace(go.Scatter(
                x=merged["date"], y=merged["official_temp_mean"],
                mode="lines+markers", line=dict(color="#fbbf24", dash="dash", width=2.5),
                marker=dict(size=7), name="Open-Meteo official"))
    fig.update_layout(yaxis_title="Mean temperature (°C)", xaxis_title="")
    st.plotly_chart(style_fig(fig, height=380), use_container_width=True)

    show = fc.copy()
    show["date"] = show["date"].dt.strftime("%a, %d %b %Y")
    show = show.rename(columns={"date": "Date", "predicted_temp": "Predicted (°C)",
                                "lower": "Lower (°C)", "upper": "Upper (°C)"})
    st.dataframe(show[["Date", "Predicted (°C)", "Lower (°C)", "Upper (°C)"]],
                 use_container_width=True, hide_index=True)
    st.caption("Intervals are calibrated on a held-out slice of the training data "
               "(residual quantiles), so about 80% of actual temperatures should "
               "fall inside the band if weather behaves like the recent past.")

# ============================================================== INSIGHTS ===
with tab_insights:
    st.markdown('<div class="section-title">🏆 Leaderboard — next-day temperature, test year '
                f"({METRICS['test_period'][0]} to {METRICS['test_period'][1]})</div>",
                unsafe_allow_html=True)
    lb = pd.DataFrame(METRICS["leaderboard"])
    st.dataframe(lb, use_container_width=True, hide_index=True)
    st.write(f"**Winner: {METRICS['winner']}** — chosen by cross-validation MAE "
             f"({METRICS['winner_cv_mae']} °C) on the training years, *before* "
             "looking at the test year. Baselines are shown first in the table "
             "so every ML model has to earn its place against them.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">🎯 Predicted vs actual (Indore, test year)</div>',
                    unsafe_allow_html=True)
        st.image(os.path.join(ROOT, "reports", "figures", "pred_vs_actual.png"))
        st.markdown('<div class="section-title">📉 Error grows with horizon</div>',
                    unsafe_allow_html=True)
        st.image(os.path.join(ROOT, "reports", "figures", "error_by_horizon.png"))
    with col2:
        st.markdown('<div class="section-title">🧠 Feature importance</div>',
                    unsafe_allow_html=True)
        st.image(os.path.join(ROOT, "reports", "figures", "feature_importance.png"))
        if METRICS.get("shap_summary_generated"):
            st.markdown('<div class="section-title">🔍 SHAP summary</div>',
                        unsafe_allow_html=True)
            st.image(os.path.join(ROOT, "reports", "figures", "shap_summary.png"))
            st.caption("Each dot is one day. Features are ordered by impact; "
                       "dots to the right pushed the predicted temperature up, "
                       "dots to the left pushed it down. Recent temperatures "
                       "(lag/rolling features) dominate — weather has strong "
                       "short-term momentum, and the seasonal cycle does the rest.")

    st.markdown('<div class="section-title">⚠️ Where does the model fail?</div>',
                unsafe_allow_html=True)
    st.write(METRICS["error_analysis"]["plain_language"])
    st.image(os.path.join(ROOT, "reports", "figures", "mae_by_month.png"))

    st.markdown(f'<div class="section-title">🌧️ Rain tomorrow — classification '
                f"({METRICS['rain_classification']['winner']})</div>",
                unsafe_allow_html=True)
    rc = METRICS["rain_classification"]
    m1, m2, m3, m4, m5 = st.columns(5)
    kpi_card(m1, "Accuracy", f"{rc['accuracy']}", "overall correct")
    kpi_card(m2, "Precision", f"{rc['precision']}", "of rain calls, right")
    kpi_card(m3, "Recall", f"{rc['recall']}", "of rain days, caught")
    kpi_card(m4, "F1 score", f"{rc['f1']}", "precision–recall balance")
    kpi_card(m5, "ROC-AUC", f"{rc['roc_auc']}", "ranking quality")
    st.image(os.path.join(ROOT, "reports", "figures", "confusion_matrix.png"), width=420)
    st.caption(f"Rain days are only about {rc['positive_rate_test'] * 100:.0f}% of all "
               "days, so accuracy alone would be misleading — F1 and ROC-AUC are "
               "the honest measures here. Class imbalance was handled with "
               "class weights during training.")

# ================================================================ DATA =====
with tab_data:
    dcity = st.selectbox("📍 City", METADATA["cities"],
                         index=METADATA["cities"].index("Indore"), key="explorer_city")
    d = HISTORY[HISTORY["city"] == dcity].sort_values("date")

    st.markdown('<div class="section-title">🌡️ Temperature trend (daily mean + 30-day average)</div>',
                unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["temp_mean"], mode="lines",
                             line=dict(color="rgba(147,180,240,0.45)", width=1),
                             name="Daily mean"))
    fig.add_trace(go.Scatter(x=d["date"], y=d["temp_mean"].rolling(30).mean(),
                             mode="lines", line=dict(color="#38bdf8", width=2.5),
                             name="30-day average"))
    fig.update_layout(yaxis_title="°C")
    st.plotly_chart(style_fig(fig, height=360), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">🍂 Seasonality — average temperature by month</div>',
                    unsafe_allow_html=True)
        monthly = d.copy()
        monthly["month"] = monthly["date"].dt.month
        seasonal = monthly.groupby("month")["temp_mean"].mean().reset_index()
        seasonal["month_name"] = seasonal["month"].apply(
            lambda m: pd.Timestamp(2025, m, 1).strftime("%b"))
        bar = px.bar(seasonal, x="month_name", y="temp_mean",
                     labels={"temp_mean": "Mean temp (°C)", "month_name": ""},
                     color="temp_mean",
                     color_continuous_scale=["#38bdf8", "#fbbf24", "#f97316"])
        bar.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(bar, height=340), use_container_width=True)
    with col2:
        st.markdown('<div class="section-title">🔗 Correlation heatmap</div>',
                    unsafe_allow_html=True)
        numeric = ["temp_max", "temp_min", "temp_mean", "humidity", "pressure",
                   "wind_speed_max", "precipitation", "cloud_cover"]
        corr = d[numeric].corr()
        heat = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1)
        st.plotly_chart(style_fig(heat, height=340), use_container_width=True)

    st.markdown('<div class="section-title">🧾 Raw data (latest 100 days)</div>',
                unsafe_allow_html=True)
    st.dataframe(d.tail(100).iloc[::-1], use_container_width=True, hide_index=True)

st.markdown('<div class="footer">🌦️ Weather Forecast · ML model: '
            f"{METADATA['winner']} · Trained on Open-Meteo data "
            f"({METADATA['data_range'][0]} → {METADATA['data_range'][1]}) · "
            "Made for the ML Lab Project</div>", unsafe_allow_html=True)
