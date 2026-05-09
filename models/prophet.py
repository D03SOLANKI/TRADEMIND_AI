"""
TradeMind AI — Prophet Model
==============================
Prophet expects tz-naive 'ds' column.
Scaled values used so metrics are near 0-1.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data.loader import future_dates
from utils.evaluation import compute_metrics_normalized
from config.settings import MIN_PROPHET

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


def forecast(df: pd.DataFrame, horizon: int) -> dict:
    if not PROPHET_AVAILABLE:
        raise ImportError("Run: pip install prophet")
    if len(df) < MIN_PROPHET:
        raise ValueError(f"Need ≥{MIN_PROPHET} rows. Got {len(df)}.")

    close  = df["Close"].values.astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close.reshape(-1, 1)).flatten()

    # Build Prophet df with tz-naive ds
    dates_col = pd.to_datetime(df["Date"])
    if dates_col.dt.tz is not None:
        dates_col = dates_col.dt.tz_localize(None)
    dates_col = dates_col.dt.normalize()

    pdf = pd.DataFrame({"ds": dates_col, "y": scaled})
    pdf = pdf.dropna()

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_mode="multiplicative",
    )
    model.fit(pdf)

    # Forecast
    future_pdf   = model.make_future_dataframe(periods=horizon)
    forecast_pdf = model.predict(future_pdf)

    fc_scaled = forecast_pdf["yhat"].iloc[-horizon:].values
    fc_prices = scaler.inverse_transform(fc_scaled.reshape(-1, 1)).flatten()

    # Eval
    eval_n  = min(30, max(5, len(pdf) // 5))
    hist    = forecast_pdf["yhat"].iloc[:-horizon].tail(eval_n).values
    true_sc = scaled[-eval_n:]

    # Compute percentage-based metrics on real prices, not scaled values.
    pred_px = scaler.inverse_transform(np.array(hist[-eval_n:]).reshape(-1, 1)).flatten()
    true_px = scaler.inverse_transform(np.array(true_sc).reshape(-1, 1)).flatten()
    metrics = compute_metrics_normalized(true_px, pred_px, reference_values=close)

    fdates  = future_dates(df, horizon)

    return {
        "forecast_df": pd.DataFrame({"Date": fdates, "Forecast": fc_prices}),
        "metrics":     metrics,
        "model_name":  "Prophet",
        "available":   True,
    }
