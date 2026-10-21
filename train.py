"""
train.py
--------
Single entry point that retrains EVERYTHING end to end:

  1. Load (or download) the daily weather dataset.
  2. Build leakage-safe supervised datasets for horizons 1..7 days.
  3. Tune temperature regressors with TimeSeriesSplit CV; baselines first.
  4. Pick the winner by cross-validation MAE (not test MAE -- honest).
  5. Calibrate prediction intervals on a held-out slice of the train set.
  6. Train one direct model per forecast horizon (1..7 days).
  7. Tune rain classifiers (imbalance-aware); evaluate on the test year.
  8. Error analysis, feature importance, SHAP, figures.
  9. Save models + metrics.json + metadata.json.

Run:  python train.py
"""

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import evaluate as ev
from src.data import clean_daily, load_daily
from src.features import (
    FEATURE_COLUMNS, build_supervised, climatology_table,
    seasonal_baseline_predict, split_time_based,
)
from src.models import get_classifiers, get_regressors, tune

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
REPORTS_DIR = os.path.join(ROOT, "reports")
X_COLS = FEATURE_COLUMNS + ["city"]
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Summer", 4: "Summer",
           5: "Summer", 6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
           10: "Post-monsoon", 11: "Post-monsoon"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def calibrate_interval(fit_fn, X_fit, y_fit, X_cal, y_cal):
    """Fit on X_fit, return residual quantiles (q10, q90) from X_cal."""
    model = fit_fn()
    model.fit(X_fit, y_fit)
    resid = y_cal - model.predict(X_cal)
    return float(np.quantile(resid, 0.10)), float(np.quantile(resid, 0.90))


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------ data
    log("Loading daily weather data ...")
    daily = clean_daily(load_daily())
    log(f"Daily dataset: {len(daily)} rows, "
        f"{daily['date'].min().date()} to {daily['date'].max().date()}")

    sup1 = build_supervised(daily, horizon=1)
    train, test, cutoff = split_time_based(sup1, test_days=365)
    log(f"Train rows: {len(train)} | Test rows: {len(test)} | cutoff {cutoff.date()}")
    X_train, y_train = train[X_COLS], train["target_temp"]
    X_test, y_test = test[X_COLS], test["target_temp"]

    clim = climatology_table(daily[daily["date"] <= cutoff])

    # ------------------------------------------------------------- baselines
    leaderboard = []
    pred_persist = test["temp_mean_lag1"].to_numpy()
    leaderboard.append({"model": "Baseline: Persistence (tomorrow = today)",
                        **ev.regression_metrics(y_test, pred_persist)})
    pred_seasonal = seasonal_baseline_predict(clim, test["city"], test["target_date"])
    leaderboard.append({"model": "Baseline: Seasonal average (climatology)",
                        **ev.regression_metrics(y_test, pred_seasonal)})

    # ------------------------------------------------- tune temperature models
    log("Tuning temperature regressors (TimeSeriesSplit CV) ...")
    regressors = get_regressors(FEATURE_COLUMNS)
    tuned = {}
    for name, (pipe, grid) in regressors.items():
        t0 = time.time()
        search = tune(pipe, grid, X_train, y_train, scoring="neg_mean_absolute_error")
        tuned[name] = search
        log(f"  {name}: CV MAE {-search.best_score_:.3f}  best={search.best_params_} "
            f"({time.time() - t0:.0f}s)")

    winner_name = min(tuned, key=lambda n: -tuned[n].best_score_)
    winner = tuned[winner_name].best_estimator_
    log(f"Winner by CV: {winner_name}")

    test_pred = {}
    for name, search in tuned.items():
        pred = search.best_estimator_.predict(X_test)
        test_pred[name] = pred
        leaderboard.append({"model": name, **ev.regression_metrics(y_test, pred)})
    leaderboard_df = pd.DataFrame(leaderboard).sort_values("MAE").reset_index(drop=True)

    # ------------------------------------- prediction intervals (calibration)
    log("Calibrating prediction intervals ...")
    cal_cut = train["target_date"].max() - pd.Timedelta(days=365)
    fit_part = train[train["target_date"] <= cal_cut]
    cal_part = train[train["target_date"] > cal_cut]
    best_params = tuned[winner_name].best_params_
    stripped_params = {k.replace("model__", ""): v for k, v in best_params.items()}

    def _fresh_winner():
        """A new, unfitted copy of the winner pipeline with its best params."""
        pipe = get_regressors(FEATURE_COLUMNS)[winner_name][0]
        pipe.named_steps["model"].set_params(**stripped_params)
        return pipe

    q10, q90 = calibrate_interval(
        lambda: _fresh_winner(), fit_part[X_COLS], fit_part["target_temp"],
        cal_part[X_COLS], cal_part["target_temp"])
    log(f"  80% interval offsets: [{q10:.2f}, {q90:.2f}] °C")

    # ------------------------------------------------------- horizon models
    log("Training direct models for horizons 1..7 ...")
    horizon_models, horizon_rows, horizon_intervals = {}, [], {}
    for h in range(1, 8):
        sup_h = build_supervised(daily, horizon=h)
        tr_h, te_h, _ = split_time_based(sup_h, test_days=365)
        model_h = _fresh_winner()
        model_h.fit(tr_h[X_COLS], tr_h["target_temp"])
        horizon_models[h] = model_h

        pred_h = model_h.predict(te_h[X_COLS])
        m = ev.regression_metrics(te_h["target_temp"], pred_h)
        horizon_rows.append({"model": winner_name, "horizon": h, **m})
        horizon_rows.append({
            "model": "Persistence", "horizon": h,
            **ev.regression_metrics(te_h["target_temp"], te_h["temp_mean_lag1"])})
        horizon_rows.append({
            "model": "Seasonal average", "horizon": h,
            **ev.regression_metrics(
                te_h["target_temp"],
                seasonal_baseline_predict(clim, te_h["city"], te_h["target_date"]))})

        cal_cut_h = tr_h["target_date"].max() - pd.Timedelta(days=365)
        fp, cp = tr_h[tr_h["target_date"] <= cal_cut_h], tr_h[tr_h["target_date"] > cal_cut_h]
        lo, hi = calibrate_interval(lambda: _fresh_winner(), fp[X_COLS], fp["target_temp"],
                                    cp[X_COLS], cp["target_temp"])
        horizon_intervals[h] = [round(lo, 2), round(hi, 2)]
    horizon_df = pd.DataFrame(horizon_rows)
    ev.fig_error_by_horizon(horizon_df)

    # ------------------------------------------------------------ rain model
    log("Tuning rain classifiers ...")
    rain_train = train.dropna(subset=["target_rain"])
    rain_test = test.dropna(subset=["target_rain"])
    Xr_tr, yr_tr = rain_train[X_COLS], rain_train["target_rain"].astype(int)
    Xr_te, yr_te = rain_test[X_COLS], rain_test["target_rain"].astype(int)
    classifiers = get_classifiers(FEATURE_COLUMNS, yr_tr)
    tuned_cls = {}
    for name, (pipe, grid) in classifiers.items():
        t0 = time.time()
        search = tune(pipe, grid, Xr_tr, yr_tr, scoring="f1")
        tuned_cls[name] = search
        log(f"  {name}: CV F1 {search.best_score_:.3f} ({time.time() - t0:.0f}s)")
    rain_winner_name = max(tuned_cls, key=lambda n: tuned_cls[n].best_score_)
    rain_model = tuned_cls[rain_winner_name].best_estimator_
    rain_pred = rain_model.predict(Xr_te)
    rain_proba = rain_model.predict_proba(Xr_te)[:, 1]
    rain_metrics = ev.classification_metrics(yr_te, rain_pred, rain_proba)
    rain_metrics["winner"] = rain_winner_name
    rain_metrics["positive_rate_test"] = round(float(yr_te.mean()), 3)
    log(f"Rain winner: {rain_winner_name} | test F1 {rain_metrics['f1']} "
        f"AUC {rain_metrics['roc_auc']}")
    ev.fig_confusion_matrix(rain_metrics["confusion_matrix"])

    # ------------------------------------------- importance, SHAP, errors
    log("Feature importance + SHAP + error analysis ...")
    from sklearn.inspection import permutation_importance
    sample_idx = np.random.RandomState(42).choice(len(X_test), size=min(1500, len(X_test)),
                                                  replace=False)
    perm = permutation_importance(winner, X_test.iloc[sample_idx], y_test.iloc[sample_idx],
                                  n_repeats=5, random_state=42,
                                  scoring="neg_mean_absolute_error")
    importance_df = (pd.DataFrame({"feature": X_COLS, "importance": perm.importances_mean})
                     .sort_values("importance", ascending=False).reset_index(drop=True))
    ev.fig_feature_importance(importance_df[importance_df["feature"] != "city"],
                              title=f"Permutation importance — {winner_name}")

    shap_done = False
    try:
        import shap
        est = winner.named_steps["model"]
        Xs = X_test.iloc[sample_idx[:500]]
        Xt = winner.named_steps["prep"].transform(Xs)
        feat_names = list(winner.named_steps["prep"].get_feature_names_out())
        Xt_df = pd.DataFrame(Xt, columns=feat_names)
        if winner_name in ("RandomForest", "XGBoost"):
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(Xt_df)
            ev.fig_shap_summary(sv, Xt_df)
            shap_done = True
        elif winner_name == "Ridge":
            explainer = shap.LinearExplainer(est, Xt_df)
            sv = explainer.shap_values(Xt_df)
            ev.fig_shap_summary(sv, Xt_df)
            shap_done = True
    except Exception as e:  # SHAP is a bonus; never fail training over it
        log(f"  SHAP skipped: {e}")

    winner_pred_test = test_pred[winner_name]
    resid = pd.DataFrame({
        "target_date": test["target_date"], "city": test["city"],
        "actual": y_test.to_numpy(), "pred": winner_pred_test,
    })
    resid["abs_err"] = (resid["actual"] - resid["pred"]).abs()
    resid["month"] = resid["target_date"].dt.month
    monthly = (resid.groupby("month")["abs_err"].mean().reset_index()
               .rename(columns={"abs_err": "MAE"}))
    monthly["month_name"] = monthly["month"].apply(lambda m: pd.Timestamp(2025, m, 1).strftime("%b"))
    ev.fig_mae_by_month(monthly)
    resid["season"] = resid["month"].map(SEASONS)
    season_mae = resid.groupby("season")["abs_err"].mean().round(3).to_dict()
    extreme = resid[resid["actual"] >= 35]
    extreme_mae = round(float(extreme["abs_err"].mean()), 3) if len(extreme) else None

    indore = test[test["city"] == "Indore"].sort_values("target_date")
    ev.fig_pred_vs_actual(indore["target_date"], indore["target_temp"],
                          winner.predict(indore[X_COLS]),
                          f"{winner_name} — predicted vs actual, Indore (test year)")

    worst_month = monthly.loc[monthly["MAE"].idxmax(), "month_name"]
    best_month = monthly.loc[monthly["MAE"].idxmin(), "month_name"]
    error_analysis = {
        "mae_by_month": {r.month_name: round(r.MAE, 3) for r in monthly.itertuples()},
        "mae_by_season": season_mae,
        "mae_extreme_heat_days_actual_ge_35C": extreme_mae,
        "plain_language": (
            f"The model is most accurate in {best_month} and least accurate in "
            f"{worst_month}. Errors are largest during the monsoon and in periods of "
            f"rapid weather change, because day-to-day temperature then depends on "
            f"rain and cloud events that yesterday's readings cannot fully reveal. "
            f"On extreme-heat days (actual mean >= 35°C) the MAE is {extreme_mae}°C. "
            f"Season-wise MAE: {season_mae}."
        ),
    }

    # ------------------------------------------------------------------ save
    joblib.dump(winner, os.path.join(MODELS_DIR, "best_regressor.joblib"), compress=3)
    joblib.dump(horizon_models, os.path.join(MODELS_DIR, "horizon_models.joblib"), compress=3)
    joblib.dump(rain_model, os.path.join(MODELS_DIR, "rain_classifier.joblib"), compress=3)
    joblib.dump(clim, os.path.join(MODELS_DIR, "climatology.joblib"), compress=3)

    metrics = {
        "task": "next-day mean temperature (°C), shared model for 5 cities",
        "test_period": [str(test["target_date"].min().date()),
                        str(test["target_date"].max().date())],
        "winner": winner_name,
        "winner_cv_mae": round(float(-tuned[winner_name].best_score_), 3),
        "winner_best_params": {k: str(v) for k, v in best_params.items()},
        "leaderboard": leaderboard_df.to_dict("records"),
        "prediction_interval_80_offsets_c": [round(q10, 2), round(q90, 2)],
        "horizon_mae": horizon_df.to_dict("records"),
        "horizon_intervals_80": horizon_intervals,
        "rain_classification": rain_metrics,
        "feature_importance_top15": importance_df.head(15).to_dict("records"),
        "shap_summary_generated": shap_done,
        "error_analysis": error_analysis,
    }
    with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    metadata = {
        "feature_columns": X_COLS,
        "cities": sorted(daily["city"].unique().tolist()),
        "winner": winner_name,
        "rain_winner": rain_winner_name,
        "interval_offsets_h1": [round(q10, 2), round(q90, 2)],
        "horizon_intervals_80": horizon_intervals,
        "data_range": [str(daily["date"].min().date()), str(daily["date"].max().date())],
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    log("DONE.")
    print("\n=== LEADERBOARD (test year, next-day temperature) ===")
    print(leaderboard_df.to_string(index=False))
    print(f"\nWinner: {winner_name} | 80% interval: [{q10:.2f}, {q90:.2f}] °C")
    print(f"Rain ({rain_winner_name}): F1 {rain_metrics['f1']}, AUC {rain_metrics['roc_auc']}")


if __name__ == "__main__":
    main()
