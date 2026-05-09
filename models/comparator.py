"""
TradeMind AI — Model Comparison Engine
========================================
Runs all available models, picks the best by RMSE.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

from models import arima as arima_model
from models import prophet as prophet_model
from models import lstm as lstm_model
from models import garch as garch_model
from models import xgboost_model as xgboost_model
from config.settings import ENABLE_LSTM, ENABLE_PROPHET


def run_all_models(df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
    """
    Run ARIMA (always), Prophet (if enabled/installed), LSTM (if enabled/installed).
    Returns:
        results    : {model_name: result_dict}
        errors     : {model_name: error_string}
        best_model : name with lowest RMSE
        comparison : pd.DataFrame sorted by RMSE
    """
    results, errors = {}, {}

    for name, mod, enabled in [
        ("ARIMA",   arima_model,   True),
        ("Prophet", prophet_model, ENABLE_PROPHET),
        ("LSTM",    lstm_model,    ENABLE_LSTM),
        ("GARCH",   garch_model,   True),
        ("XGBoost", xgboost_model, True),
    ]:
        if not enabled:
            continue
        try:
            results[name] = mod.forecast(df, horizon)
        except Exception as e:
            errors[name] = str(e)

    rows = []
    for n, r in results.items():
        metrics = r.get("metrics", {}) if isinstance(r, dict) else {}
        rows.append({
            "Model": n,
            "RMSE": round(metrics.get("RMSE (norm)", np.nan), 4) if metrics.get("RMSE (norm)") is not None else np.nan,
            "MAE": round(metrics.get("MAE (norm)", np.nan), 4) if metrics.get("MAE (norm)") is not None else np.nan,
            "Accuracy": round(metrics.get("Accuracy", np.nan), 4) if metrics.get("Accuracy") is not None else np.nan,
            "ROC-AUC": round(metrics.get("ROC-AUC", np.nan), 4) if metrics.get("ROC-AUC") is not None else np.nan,
        })
    comparison = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True) if rows else pd.DataFrame()
    best_model = comparison.iloc[0]["Model"] if not comparison.empty else None

    return {
        "results":    results,
        "errors":     errors,
        "best_model": best_model,
        "comparison": comparison,
    }
