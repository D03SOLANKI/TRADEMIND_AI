"""
TradeMind AI — GARCH Volatility Forecasting
===========================================
Provides a lightweight GARCH(1,1) volatility engine with regime detection.
"""

import numpy as np
import pandas as pd
from data.loader import future_dates
from utils.evaluation import compute_metrics_normalized
from config.settings import MIN_ARIMA

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False


def _estimate_garch_params(returns: pd.Series):
    eps2 = returns.pow(2)
    var = float(eps2.mean())
    best_cost = np.inf
    best_params = (max(var * 0.01, 1e-8), 0.1, 0.85, np.full(len(eps2), var))

    for alpha in [0.05, 0.075, 0.1, 0.125, 0.15, 0.2]:
        for beta in [0.7, 0.75, 0.8, 0.85, 0.9, 0.94]:
            if alpha + beta >= 0.995:
                continue
            omega = var * (1 - alpha - beta)
            h = np.empty(len(eps2))
            h[0] = var
            for t in range(1, len(eps2)):
                h[t] = omega + alpha * eps2.iloc[t - 1] + beta * h[t - 1]
            cost = float(np.mean((eps2.values - h) ** 2))
            if cost < best_cost:
                best_cost = cost
                best_params = (omega, alpha, beta, h)

    return best_params


def _garch_forecast_variance(last_var: float, omega: float, alpha: float, beta: float, horizon: int):
    forecasts = []
    h = float(last_var)
    for _ in range(horizon):
        h = omega + (alpha + beta) * h
        forecasts.append(h)
    return np.array(forecasts, dtype=float)


def _vol_regime(vol: float) -> str:
    if vol < 0.012:
        return "Low volatility"
    if vol < 0.03:
        return "Moderate volatility"
    return "High volatility"


def forecast(df: pd.DataFrame, horizon: int) -> dict:
    if len(df) < MIN_ARIMA:
        raise ValueError(f"Need ≥{MIN_ARIMA} rows. Got {len(df)}.")

    # Use log returns for a more stable volatility process.
    returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    if returns.empty:
        raise ValueError("Not enough valid returns to fit GARCH.")

    annualize = np.sqrt(252.0)

    if ARCH_AVAILABLE:
        model = arch_model(returns * 100.0, mean="Zero", vol="Garch", p=1, q=1, dist="normal")
        res = model.fit(disp="off")
        cond_vol = (res.conditional_volatility / 100.0 * annualize).reindex(returns.index)
        forecast_variance = res.forecast(horizon=horizon, reindex=False).variance.iloc[-1].values / 10000.0
        forecast_vol = np.sqrt(np.maximum(forecast_variance, 0.0)) * annualize
        alpha = float(res.params.get("alpha[1]", 0.1))
        beta = float(res.params.get("beta[1]", 0.85))
        omega = float(res.params.get("omega", returns.var() * (1 - alpha - beta)))
    else:
        omega, alpha, beta, h_series = _estimate_garch_params(returns)
        cond_vol = np.sqrt(np.maximum(pd.Series(h_series, index=returns.index), 0.0)) * annualize
        forecast_vol = np.sqrt(np.maximum(_garch_forecast_variance((cond_vol.iloc[-1] / annualize) ** 2, omega, alpha, beta, horizon), 0.0)) * annualize

    # Smooth the path and anchor it to recent realized volatility to avoid noisy spikes.
    forecast_vol = pd.Series(forecast_vol).ewm(alpha=0.45, adjust=False).mean().values
    recent_realized_vol = float((returns.rolling(14).std() * annualize).dropna().iloc[-1]) if len(returns) >= 14 else float(np.nanmean(cond_vol))
    forecast_vol = 0.7 * forecast_vol + 0.3 * recent_realized_vol

    fdates = future_dates(df, horizon)
    upper = forecast_vol * 1.15
    lower = np.maximum(forecast_vol * 0.85, 0.0)

    vol_forecast = {
        "7d": float(np.mean(forecast_vol[:7])) if len(forecast_vol) >= 7 else float(np.mean(forecast_vol)),
        "14d": float(np.mean(forecast_vol[:14])) if len(forecast_vol) >= 14 else float(np.mean(forecast_vol)),
        "30d": float(np.mean(forecast_vol[:30])) if len(forecast_vol) >= 30 else float(np.mean(forecast_vol)),
    }

    regime = _vol_regime(float(cond_vol.iloc[-1]))
    warnings = []
    if regime == "High volatility":
        warnings.append("High volatility regime detected — signals are less reliable.")
    elif regime == "Moderate volatility":
        warnings.append("Moderate volatility environment — monitor risk closely.")
    else:
        warnings.append("Low volatility environment — tighter risk control may be appropriate.")

    hist_vol = (returns.rolling(14).std() * annualize).dropna()
    metrics = {}
    if len(hist_vol) >= 5:
        # Score against a persistence baseline: compare the forecast path to the
        # latest realized annualized volatility level. This is the appropriate
        # reference for a volatility forecaster and keeps the comparison stable.
        recent_vol = float(hist_vol.iloc[-1])
        actual_eval = np.full(len(forecast_vol), recent_vol, dtype=float)
        pred_eval = np.asarray(forecast_vol, dtype=float)
        metrics = compute_metrics_normalized(actual_eval, pred_eval, reference_values=hist_vol.values)
    else:
        metrics = {"RMSE (norm)": np.nan, "MAE (norm)": np.nan, "MAPE (%)": np.nan}

    return {
        "forecast_df": pd.DataFrame({
            "Date": fdates,
            "Forecast": forecast_vol,
            "Upper": upper,
            "Lower": lower,
        }),
        "metrics": metrics,
        "model_name": "GARCH",
        "conditional_volatility": pd.DataFrame({"Date": returns.index, "Volatility": cond_vol}),
        "vol_forecast": vol_forecast,
        "regime": regime,
        "warnings": warnings,
        "available": True,
    }
