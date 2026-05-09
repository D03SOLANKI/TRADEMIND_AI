"""
TradeMind AI — XGBoost Directional Prediction Model
===================================================
Provides a binary up/down classifier that uses technical indicators,
volatility regimes, and optional sentiment features to predict the next
price direction and probability of a bullish move.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from data.loader import future_dates
from features.indicators import add_all
from config.settings import MIN_ARIMA

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

MIN_XGB = 20
FEATURE_COLUMNS = [
    "RSI", "MACD", "MACD_sig", "MACD_hist",
    "BB_pct", "BB_width", "ATR",
    "Vol7", "Vol30", "Ret",
    "Stoch_K", "Stoch_D", "Volume",
    "GARCH_Regime",
    "polarity", "subjectivity",
]


def _regime_code(vol: float) -> int:
    if pd.isna(vol):
        return 1
    if vol < 0.012:
        return 0
    if vol < 0.03:
        return 1
    return 2


def _build_feature_matrix(df: pd.DataFrame, sentiment_features: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    df = add_all(df)

    if "Ret" not in df.columns:
        df["Ret"] = df["Close"].pct_change()
    df["Ret_lag1"] = df["Ret"].shift(1)
    df["Ret_lag2"] = df["Ret"].shift(2)
    df["Ret_lag3"] = df["Ret"].shift(3)
    df["Mom3"] = df["Close"].pct_change(periods=3)
    df["Mom5"] = df["Close"].pct_change(periods=5)
    df["Vol_ratio"] = df["Vol7"] / (df["Vol30"] + 1e-9) if "Vol7" in df.columns and "Vol30" in df.columns else np.nan

    # Trend spreads
    if "SMA30" in df.columns:
        df["SMA_spread"] = (df["Close"] - df["SMA30"]) / (df["SMA30"] + 1e-9)
    else:
        df["SMA_spread"] = np.nan
    if "EMA26" in df.columns:
        df["EMA_spread"] = (df["Close"] - df["EMA26"]) / (df["EMA26"] + 1e-9)
    else:
        df["EMA_spread"] = np.nan
    if "VWAP" in df.columns:
        df["VWAP_spread"] = (df["Close"] - df["VWAP"]) / (df["VWAP"] + 1e-9)
    else:
        df["VWAP_spread"] = np.nan
    if "GARCH_Regime" not in df.columns:
        df["GARCH_Regime"] = df["Vol30"].apply(_regime_code) if "Vol30" in df.columns else 1

    sentiment_features = sentiment_features or {}
    polarity = float(sentiment_features.get("polarity", 0.0))
    subjectivity = float(sentiment_features.get("subjectivity", 0.0))
    if "polarity" not in df.columns:
        df["polarity"] = polarity
    else:
        df["polarity"] = df["polarity"].fillna(polarity)
    if "subjectivity" not in df.columns:
        df["subjectivity"] = subjectivity
    else:
        df["subjectivity"] = df["subjectivity"].fillna(subjectivity)

    return df


def _prepare_dataset(df: pd.DataFrame, sentiment_features: dict | None = None, target_horizon: int = 1) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = _build_feature_matrix(df, sentiment_features=sentiment_features)
    valid = df.dropna(subset=FEATURE_COLUMNS + ["Close"]).copy()
    if len(valid) < MIN_XGB:
        raise ValueError(f"Need ≥{MIN_XGB} rows of valid XGBoost features. Got {len(valid)}.")

    step = max(1, int(target_horizon))
    y = (valid["Close"].shift(-step) > valid["Close"]).astype(int)
    valid = valid.iloc[:-step].copy()
    y = y.iloc[:-step]

    if len(y.unique()) < 2:
        raise ValueError("Not enough up/down direction variety to train XGBoost.")

    X = valid[FEATURE_COLUMNS].copy()
    return X, y, valid


def _train_classifier(X: pd.DataFrame, y: pd.Series, use_tuning: bool = False):
    """Train classifier using common training wrapper (early stopping).

    `use_tuning` currently reserved for future auto-search; wrapper always
    uses an 80/20 split for an internal validation set used for early stopping.
    """
    from utils.training import train_classifier_with_early_stopping

    split = max(1, int(len(X) * 0.8))
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    model, scaler = train_classifier_with_early_stopping(X_train, y_train, X_val, y_val, use_tuning=use_tuning)

    # Tune decision threshold on validation to maximize accuracy.
    best_thr = 0.5
    if hasattr(model, "predict_proba") and len(X_val) > 0:
        probs_val = model.predict_proba(scaler.transform(X_val))[:, 1]
        best_acc = -1.0
        for thr in np.linspace(0.35, 0.65, 31):
            pred_val = (probs_val >= thr).astype(int)
            acc = accuracy_score(y_val, pred_val)
            if acc > best_acc:
                best_acc = acc
                best_thr = float(thr)

    return model, scaler, best_thr


def _evaluate_model(model, scaler, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5) -> dict:
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(X_scaled))
    pred = (probs >= threshold).astype(int)

    acc = accuracy_score(y, pred)
    prec = precision_score(y, pred, zero_division=0)
    rec = recall_score(y, pred, zero_division=0)
    f1 = f1_score(y, pred, zero_division=0)
    roc = roc_auc_score(y, probs) if len(np.unique(y)) == 2 else np.nan
    cm = confusion_matrix(y, pred, labels=[0, 1])

    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "ROC-AUC": float(roc) if not np.isnan(roc) else np.nan,
        "Confusion Matrix": pd.DataFrame(
            cm,
            index=["Down", "Up"],
            columns=["Pred Down", "Pred Up"],
        ),
        "Predictions": pred,
        "Probabilities": probs,
    }


def _feature_importance(model, scaler, X: pd.DataFrame, y: pd.Series, feature_names: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = np.array(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.array(model.coef_).ravel())
    else:
        values = np.zeros(len(feature_names), dtype=float)

    # Fallback for models without native importances (e.g., HistGradientBoostingClassifier)
    # or when native importances are all near-zero.
    if values.size != len(feature_names) or not np.isfinite(values).all() or float(np.abs(values).sum()) < 1e-12:
        X_scaled = scaler.transform(X)
        try:
            score = "roc_auc" if len(np.unique(y)) == 2 else "accuracy"
            perm = permutation_importance(
                model,
                X_scaled,
                y,
                n_repeats=4,
                random_state=42,
                scoring=score,
            )
            values = np.maximum(perm.importances_mean, 0.0)
        except Exception:
            values = np.zeros(len(feature_names), dtype=float)

    df = pd.DataFrame({"Feature": feature_names, "Importance": values})
    df = df.sort_values("Importance", ascending=False).reset_index(drop=True)
    return df


def forecast(df: pd.DataFrame, horizon: int, sentiment_features: dict | None = None) -> dict:
    if len(df) < MIN_XGB:
        raise ValueError(f"Need ≥{MIN_XGB} rows. Got {len(df)}.")

    # Keep one-step directional target for stable ranking metrics.
    X, y, valid = _prepare_dataset(df, sentiment_features=sentiment_features, target_horizon=1)
    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    if len(X_test) == 0:
        X_test, y_test = X_train, y_train

    model, scaler, threshold = _train_classifier(X_train, y_train)
    eval_result = _evaluate_model(model, scaler, X_test, y_test, threshold=threshold)

    X_last = X.iloc[[-1]]
    X_last_scaled = scaler.transform(X_last)
    prob_up = float(model.predict_proba(X_last_scaled)[:, 1][0]) if hasattr(model, "predict_proba") else 0.0
    pred_label = "Up" if prob_up >= threshold else "Down"
    classification = "Bullish" if pred_label == "Up" else "Bearish"

    feature_importance = _feature_importance(model, scaler, X_test, y_test, list(X.columns))
    fdates = future_dates(df, horizon)
    forecast_df = pd.DataFrame({"Date": fdates, "Forecast": [prob_up * 100.0] * horizon})

    return {
        "forecast_df": forecast_df,
        "metrics": {
            "Accuracy": round(eval_result["Accuracy"], 4),
            "Precision": round(eval_result["Precision"], 4),
            "Recall": round(eval_result["Recall"], 4),
            "F1": round(eval_result["F1"], 4),
            "ROC-AUC": round(eval_result["ROC-AUC"], 4) if not np.isnan(eval_result["ROC-AUC"]) else np.nan,
        },
        "confusion_matrix": eval_result["Confusion Matrix"],
        "feature_importance": feature_importance,
        "prediction": pred_label,
        "probability": prob_up,
        "label": classification,
        "model_name": "XGBoost",
        "available": True,
        "feature_names": list(X.columns),
        "warnings": [] if XGBOOST_AVAILABLE else ["XGBoost package not installed — using sklearn fallback classifier."],
    }
