"""TradeMind AI — Evaluation Metrics"""
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_rmse_mae(y_true, y_pred):
    """Returns (rmse, mae) on raw values."""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return 0.0, 0.0
    rmse = math.sqrt(mean_squared_error(y_true[-n:], y_pred[-n:]))
    mae  = mean_absolute_error(y_true[-n:], y_pred[-n:])
    return round(rmse, 6), round(mae, 6)


def compute_metrics_normalized(y_true, y_pred, reference_values=None):
    """
    Returns normalized metrics where RMSE/MAE are close to 0-1 range.
    Uses MinMax-scaled values so metrics are scale-independent.
    Also returns MAPE (%) which is the most useful for financial data.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[-n:], y_pred[-n:]

    # Normalize by a stable price scale (mean absolute level of full history
    # when available). This keeps normalized errors interpretable and avoids
    # inflated values from tiny local windows.
    if reference_values is not None:
        ref = np.array(reference_values).flatten()
        ref = ref[np.isfinite(ref)]
    else:
        ref = y_true[np.isfinite(y_true)]

    scale = float(np.mean(np.abs(ref))) if len(ref) else 1.0
    if scale <= 1e-9:
        scale = 1.0

    rmse_raw = math.sqrt(mean_squared_error(y_true, y_pred))
    mae_raw = mean_absolute_error(y_true, y_pred)
    rmse_n = rmse_raw / scale
    mae_n = mae_raw / scale

    # MAPE — protected against near-zero true values by applying small epsilon
    eps = 1e-6
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100) if len(denom) > 0 else 0.0

    # SMAPE (symmetric MAPE)
    smape_denom = (np.abs(y_true) + np.abs(y_pred))
    smape_denom = np.where(smape_denom < eps, eps, smape_denom)
    smape = float(np.mean(np.abs(y_pred - y_true) / (smape_denom / 2.0)) * 100)

    return {
        "RMSE (norm)": round(rmse_n, 4),
        "MAE (norm)":  round(mae_n,  4),
        "MAPE (%)":    round(mape,   2),
        "SMAPE (%)":   round(smape,  2),
    }
