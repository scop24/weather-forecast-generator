"""
src/evaluate.py
---------------
Metric helpers and figure generation for the reports/ folder.
All figures are saved as PNGs that the Streamlit app displays.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
    }


def classification_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 3),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_pred_vs_actual(dates, y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(dates, y_true, label="Actual", lw=1.2)
    ax.plot(dates, y_pred, label="Predicted", lw=1.2, alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel("Mean temperature (°C)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "pred_vs_actual.png")


def fig_error_by_horizon(horizon_df):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, g in horizon_df.groupby("model"):
        ax.plot(g["horizon"], g["MAE"], marker="o", label=name)
    ax.set_xlabel("Forecast horizon (days ahead)")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("How error grows with forecast horizon (test set)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "error_by_horizon.png")


def fig_feature_importance(importance_df, title="Feature importance"):
    top = importance_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"], color="#3b82c4")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    return _save(fig, "feature_importance.png")


def fig_confusion_matrix(cm, labels=("No rain", "Rain")):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Rain classifier — confusion matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.colorbar(im, shrink=0.8)
    return _save(fig, "confusion_matrix.png")


def fig_mae_by_month(monthly_df):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(monthly_df["month_name"], monthly_df["MAE"], color="#c0563b")
    ax.set_ylabel("MAE (°C)")
    ax.set_title("Test-set error by month — where the model struggles")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "mae_by_month.png")


def fig_shap_summary(shap_values, X_sample, name="shap_summary.png"):
    import shap
    fig = plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.title("SHAP summary — impact of each feature on the prediction")
    return _save(fig, name)
