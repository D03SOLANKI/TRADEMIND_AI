"""Hyperparameter tuning helpers for classifiers and LSTM.

Provides lightweight wrappers around sklearn's search utilities and a
small grid-search loop for TensorFlow LSTM when TF is available.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


def tune_xgb_classifier(X: pd.DataFrame, y: pd.Series, n_iter: int = 20, cv: int = 3, random_state: int = 42) -> Dict[str, Any]:
    """Randomized search for XGBoost classifier. Returns dict with best estimator and params.

    Falls back to sklearn's HistGradientBoostingClassifier if xgboost is not installed.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    pipe_steps = [("scaler", StandardScaler())]
    if XGBOOST_AVAILABLE:
        base = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state, verbosity=0, n_jobs=-1)
        pipe_steps.append(("clf", base))
        param_dist = {
            "clf__n_estimators": [50, 80, 120, 200],
            "clf__max_depth": [2, 3, 4, 6],
            "clf__learning_rate": [0.01, 0.03, 0.08, 0.1],
            "clf__subsample": [0.6, 0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 0.9, 1.0],
        }
    else:
        base = HistGradientBoostingClassifier(random_state=random_state)
        pipe_steps.append(("clf", base))
        param_dist = {
            "clf__max_iter": [50, 100, 150, 200],
        }

    pipeline = Pipeline(pipe_steps)

    search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=min(n_iter, 40), cv=cv, scoring="roc_auc", random_state=random_state, n_jobs=-1)
    search.fit(X, y)
    return {"best_estimator": search.best_estimator_, "best_params": search.best_params_, "cv_results": search.cv_results_}


def tune_lstm_simple(series: np.ndarray, lookback: int = 10, grid: Optional[Dict[str, list]] = None, epochs: int = 10, verbose: int = 0) -> Dict[str, Any]:
    """Brute-force grid search for small LSTM configs. Expects 1D numpy `series` scaled to [0,1].

    Returns best config and history (val loss) summary. This is intentionally lightweight.
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow not available — install tensorflow to tune LSTM")

    if grid is None:
        grid = {
            "units": [8, 16],
            "dropout": [0.0, 0.2],
            "batch_size": [16, 32],
        }

    # Build sequences
    def make_seq(s, lb):
        X, y = [], []
        for i in range(len(s) - lb):
            X.append(s[i:i+lb])
            y.append(s[i+lb])
        return np.array(X), np.array(y)

    X, y = make_seq(series.reshape(-1,1), lookback)
    if len(X) == 0:
        raise ValueError("Not enough data for lookback")

    # reshape
    X = X.reshape(-1, lookback, 1)

    # simple 80/20 split
    split = int(len(X)*0.8)
    Xtr, Xval = X[:split], X[split:]
    ytr, yval = y[:split], y[split:]

    best = {"loss": float("inf")}

    import itertools
    for units, dropout, batch_size in itertools.product(grid["units"], grid["dropout"], grid["batch_size"]):
        tf.keras.backend.clear_session()
        model = Sequential([
            LSTM(units, input_shape=(lookback,1)),
            Dropout(dropout),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        cb = [EarlyStopping(patience=3, restore_best_weights=True)]
        hist = model.fit(Xtr, ytr, validation_data=(Xval,yval), epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=cb)
        val_loss = float(np.min(hist.history.get("val_loss", hist.history.get("loss", [float("inf")]))))
        if val_loss < best["loss"]:
            best = {"units": units, "dropout": dropout, "batch_size": batch_size, "loss": val_loss}

    return best


__all__ = ["tune_xgb_classifier", "tune_lstm_simple"]

def tune_prophet_simple(df: pd.DataFrame, horizon: int = 7, grid: dict | None = None) -> dict:
    """Lightweight grid search over Prophet hyperparameters.

    Returns best config and evaluation metrics. Expects a DataFrame with
    `Date` and `Close` columns (like `preprocess` output).
    """
    try:
        from prophet import Prophet
    except Exception:
        raise ImportError("Prophet not available — install prophet to tune it")

    from utils.evaluation import compute_metrics_normalized
    from sklearn.preprocessing import MinMaxScaler

    if grid is None:
        grid = {
            "changepoint_prior_scale": [0.01, 0.05, 0.1],
            "seasonality_mode": ["additive", "multiplicative"],
            "weekly_seasonality": [True, False],
            "yearly_seasonality": [True, False],
        }

    close = df["Close"].values.astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close.reshape(-1, 1)).flatten()

    dates_col = pd.to_datetime(df["Date"]) if "Date" in df.columns else pd.date_range(periods=len(df))
    if dates_col.dt.tz is not None:
        dates_col = dates_col.dt.tz_localize(None)
    dates_col = dates_col.dt.normalize()

    pdf = pd.DataFrame({"ds": dates_col, "y": scaled}).dropna()

    import itertools
    best = {"score": float("inf"), "config": None, "metrics": None}

    combos = list(itertools.product(grid["changepoint_prior_scale"], grid["seasonality_mode"], grid["weekly_seasonality"], grid["yearly_seasonality"]))
    for cps, smode, weekly, yearly in combos:
        try:
            model = Prophet(daily_seasonality=False, weekly_seasonality=weekly, yearly_seasonality=yearly,
                            changepoint_prior_scale=cps, seasonality_mode=smode)
            model.fit(pdf)

            future_pdf = model.make_future_dataframe(periods=horizon)
            forecast_pdf = model.predict(future_pdf)

            # Evaluate on last chunk
            eval_n = min(30, max(5, len(pdf) // 5))
            hist = forecast_pdf["yhat"].iloc[:-horizon].tail(eval_n).values
            true_sc = scaled[-eval_n:]
            metrics = compute_metrics_normalized(true_sc, hist[-eval_n:])

            # Use normalized RMSE as score
            score = metrics.get("RMSE (norm)", float("inf"))
            if score < best["score"]:
                best["score"] = score
                best["config"] = {"changepoint_prior_scale": cps, "seasonality_mode": smode, "weekly_seasonality": weekly, "yearly_seasonality": yearly}
                best["metrics"] = metrics
        except Exception:
            continue

    return best
