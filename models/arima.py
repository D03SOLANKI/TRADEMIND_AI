"""
TradeMind AI — ARIMA Model
============================
ROOT CAUSE FIX: The error 'int + datetime.datetime' happens because
statsmodels / pmdarima internally do integer arithmetic on the series index.
When that index is a DatetimeIndex, it crashes.

DEFINITIVE FIX: Feed only a plain numpy array (RangeIndex series) to ARIMA.
Dates are computed separately from safe_last_date() using timedelta only.
This completely eliminates ANY date arithmetic inside the model code.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data.loader import future_dates
from utils.evaluation import compute_metrics_normalized
from config.settings import MIN_ARIMA

try:
    from statsmodels.tsa.arima.model import ARIMA
    from pmdarima import auto_arima
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False


def forecast(df: pd.DataFrame, horizon: int) -> dict:
    """
    ARIMA forecast using RangeIndex series — no DatetimeIndex in model.

    Returns: {forecast_df, metrics, model_name, order, available}
    """
    if not ARIMA_AVAILABLE:
        raise ImportError("Run: pip install statsmodels pmdarima")

    if len(df) < MIN_ARIMA:
        raise ValueError(f"Need ≥{MIN_ARIMA} rows. Got {len(df)}.")

    # ── STEP 1: Extract values only — NO DatetimeIndex into model ──
    close = df["Close"].values.astype(float)

    # Scale to [0,1] so RMSE/MAE are near 0 (not 40000+)
    scaler  = MinMaxScaler()
    scaled  = scaler.fit_transform(close.reshape(-1, 1)).flatten()

    # RangeIndex series — statsmodels will NEVER touch any date
    series  = pd.Series(scaled)   # index = 0,1,2,... (integers)

    # ── STEP 2: Find best order ──
    auto = auto_arima(
        series, seasonal=False, suppress_warnings=True,
        error_action="ignore", stepwise=True, max_p=4, max_q=4, max_d=2,
    )
    order = auto.order

    # ── STEP 3: Fit on values (RangeIndex — safe) ──
    model  = ARIMA(series, order=order)
    result = model.fit()

    # ── STEP 4: Forecast — returns plain array, no date ops ──
    fc_scaled = np.array(result.forecast(steps=horizon)).flatten()
    fc_prices = scaler.inverse_transform(fc_scaled.reshape(-1, 1)).flatten()

    # ── STEP 5: Eval using integer indices (RangeIndex — 100% safe) ──
    eval_n   = min(30, max(5, len(series) // 5))
    # predict(start=int, end=int) is safe with RangeIndex
    pred_sc  = np.array(result.predict(
        start=len(series) - eval_n,
        end=len(series) - 1,
    )).flatten()
    true_sc  = scaled[-eval_n:]

    # Compute percentage-based metrics on real prices, not scaled values.
    pred_px = scaler.inverse_transform(pred_sc.reshape(-1, 1)).flatten()
    true_px = scaler.inverse_transform(true_sc.reshape(-1, 1)).flatten()
    metrics = compute_metrics_normalized(true_px, pred_px, reference_values=close)

    # ── STEP 6: Build future dates from safe source ONLY ──
    fdates = future_dates(df, horizon)

    return {
        "forecast_df": pd.DataFrame({"Date": fdates, "Forecast": fc_prices}),
        "metrics":     metrics,
        "model_name":  "ARIMA",
        "order":       order,
        "available":   True,
    }
