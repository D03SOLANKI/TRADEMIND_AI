"""
TradeMind AI — LSTM Model
===========================
Scaled training → normalized RMSE/MAE near 0-1.
Future dates from future_dates() only — no date arithmetic in model.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data.loader import future_dates
from utils.evaluation import compute_metrics_normalized
from config.settings import MIN_LSTM, LOOKBACK, EPOCHS, LSTM_UNITS

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from utils.training import train_lstm_with_wrapper
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def _make_sequences(data: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i: i + lookback, 0])
        y.append(data[i + lookback, 0])
    return np.array(X), np.array(y)


def forecast(df: pd.DataFrame, horizon: int) -> dict:
    if not TF_AVAILABLE:
        raise ImportError("Run: pip install tensorflow")
    if len(df) < MIN_LSTM:
        raise ValueError(f"Need ≥{MIN_LSTM} rows. Got {len(df)}.")

    lookback = min(LOOKBACK, max(1, len(df) // 2))
    close  = df["Close"].values.astype(float).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close)

    X, y = _make_sequences(scaled, lookback)
    if len(X) == 0:
        raise ValueError("Unable to build LSTM sequences from the available data.")

    split = int(len(X) * 0.8)
    split = max(1, min(split, len(X)))
    Xtr, Xte = X[:split].reshape(-1, lookback, 1), X[split:].reshape(-1, lookback, 1)
    ytr, yte  = y[:split], y[split:]

    # Use training wrapper for LSTM (provides early stopping and clean session)
    model = train_lstm_with_wrapper(Xtr, ytr, Xte if len(Xte) > 0 else None, yte if len(Xte) > 0 else None,
                                    units=LSTM_UNITS, dropout=0.2, epochs=EPOCHS, batch_size=32, patience=3)

    eval_X, eval_y = (Xte, yte) if len(Xte) > 0 else (Xtr, ytr)

    # Eval on held-out data when possible, otherwise use training set.
    pred_te = model.predict(eval_X, verbose=0).flatten()

    # Compute percentage-based metrics on real prices, not scaled values.
    pred_px = scaler.inverse_transform(pred_te.reshape(-1, 1)).flatten()
    true_px = scaler.inverse_transform(eval_y.reshape(-1, 1)).flatten()
    metrics = compute_metrics_normalized(true_px, pred_px, reference_values=close.flatten())

    # Multi-step future
    seq = scaled[-lookback:].reshape(1, lookback, 1)
    preds_sc = []
    for _ in range(horizon):
        nv = float(model.predict(seq, verbose=0)[0][0])
        preds_sc.append(nv)
        seq = np.append(seq[:, 1:, :], [[[nv]]], axis=1)

    fc_prices = scaler.inverse_transform(
        np.array(preds_sc).reshape(-1, 1)
    ).flatten()

    fdates = future_dates(df, horizon)

    return {
        "forecast_df": pd.DataFrame({"Date": fdates, "Forecast": fc_prices}),
        "metrics":     metrics,
        "model_name":  "LSTM",
        "available":   True,
    }
