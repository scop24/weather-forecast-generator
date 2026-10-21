"""
src/models.py
-------------
Model definitions and tuning helpers.

One shared model is trained for all cities, with `city` as a one-hot
encoded feature (chosen over 5 separate models: ~5x more training data
per model, one pipeline to maintain, and the model can learn patterns
shared across cities -- e.g. monsoon behaviour -- while the one-hot
feature still lets it learn each city's offset).

Every model is an sklearn Pipeline: preprocessing (one-hot city,
optional scaling) + estimator, so training and inference always apply
identical preprocessing. Tuning uses TimeSeriesSplit -- never random CV,
which would let the model "peek" at the future during validation.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

RANDOM_STATE = 42


def make_preprocessor(numeric_features, scale=False):
    """One-hot encode `city`; optionally standardise the numeric features."""
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["city"]),
    ])


def _pipe(numeric_features, estimator, scale=False):
    return Pipeline([
        ("prep", make_preprocessor(numeric_features, scale=scale)),
        ("model", estimator),
    ])


def get_regressors(numeric_features):
    """name -> (pipeline, hyperparameter grid) for temperature regression."""
    return {
        "Ridge": (
            _pipe(numeric_features, Ridge(), scale=True),
            {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        "RandomForest": (
            _pipe(numeric_features, RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
            {"model__n_estimators": [300],
             "model__max_depth": [None, 14],
             "model__min_samples_leaf": [1, 2]},
        ),
        "XGBoost": (
            _pipe(numeric_features, XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
            {"model__n_estimators": [400],
             "model__learning_rate": [0.05, 0.1],
             "model__max_depth": [4, 6]},
        ),
        "MLP": (
            _pipe(numeric_features,
                  MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400,
                               early_stopping=True, random_state=RANDOM_STATE),
                  scale=True),
            {"model__alpha": [0.01, 0.1]},
        ),
    }


def get_classifiers(numeric_features, y_train):
    """name -> (pipeline, grid) for rain classification, imbalance-aware."""
    n_pos = max(int(y_train.sum()), 1)
    n_neg = len(y_train) - n_pos
    return {
        "LogisticRegression": (
            _pipe(numeric_features,
                  LogisticRegression(class_weight="balanced", max_iter=1000),
                  scale=True),
            {"model__C": [0.1, 1.0]},
        ),
        "RandomForest": (
            _pipe(numeric_features,
                  RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                         class_weight="balanced_subsample")),
            {"model__n_estimators": [300], "model__max_depth": [None, 14]},
        ),
        "XGBoost": (
            _pipe(numeric_features,
                  XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                scale_pos_weight=n_neg / n_pos,
                                eval_metric="logloss")),
            {"model__n_estimators": [300], "model__max_depth": [4, 6],
             "model__learning_rate": [0.05, 0.1]},
        ),
        "MLP": (
            _pipe(numeric_features,
                  MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400,
                                early_stopping=True, random_state=RANDOM_STATE),
                  scale=True),
            {"model__alpha": [0.01, 0.1]},
        ),
    }


def tune(pipeline, grid, X, y, scoring, n_splits=3):
    """Grid search with TimeSeriesSplit CV. Returns fitted GridSearchCV."""
    cv = TimeSeriesSplit(n_splits=n_splits)
    search = GridSearchCV(pipeline, grid, cv=cv, scoring=scoring, n_jobs=1)
    search.fit(X, y)
    return search
