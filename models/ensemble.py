"""Simple ensemble helpers combining multiple model forecast outputs.

Each model forecast dict is expected to contain a `forecast_df` DataFrame
with `Date` and `Forecast` columns and a `metrics` mapping.
"""
from typing import List, Dict, Any
import numpy as np
import pandas as pd


def ensemble_average(forecasts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Averages numeric forecasts and aggregates metrics by mean.

    Returns an ensemble dict in the same shape as model forecast dicts.
    """
    if not forecasts:
        return {"forecast_df": pd.DataFrame(), "metrics": {}, "model_name": "Ensemble", "available": False}

    # Align by Date and average Forecast
    dfs = [f["forecast_df"].set_index("Date")["Forecast"] for f in forecasts if "forecast_df" in f and not f["forecast_df"].empty]
    if not dfs:
        return {"forecast_df": pd.DataFrame(), "metrics": {}, "model_name": "Ensemble", "available": False}

    joined = pd.concat(dfs, axis=1)
    avg = joined.mean(axis=1)
    forecast_df = avg.reset_index().rename(columns={0: "Forecast"})

    # Aggregate metrics: take numeric metrics and average them when possible
    metric_keys = set().union(*(set(f.get("metrics", {}).keys()) for f in forecasts))
    agg_metrics = {}
    for k in metric_keys:
        vals = []
        for f in forecasts:
            m = f.get("metrics", {})
            if k in m and isinstance(m[k], (int, float)):
                vals.append(float(m[k]))
        if vals:
            agg_metrics[k] = sum(vals) / len(vals)

    return {
        "forecast_df": forecast_df,
        "metrics": agg_metrics,
        "model_name": "Ensemble",
        "available": True,
    }


def ensemble_weighted_by_mape(forecasts: List[Dict[str, Any]], eps: float = 1e-6) -> Dict[str, Any]:
    """Weighted ensemble using inverse MAPE (%) as weights when available.

    If no MAPE available, falls back to simple average.
    """
    if not forecasts:
        return {"forecast_df": pd.DataFrame(), "metrics": {}, "model_name": "EnsembleWeighted", "available": False}

    # collect forecast series
    dfs = []
    model_mapes = []
    for f in forecasts:
        if "forecast_df" in f and not f["forecast_df"].empty:
            s = f["forecast_df"].set_index("Date")["Forecast"]
            dfs.append(s)
            m = f.get("metrics", {}).get("MAPE (%)")
            model_mapes.append(m if m is not None else None)

    if not dfs:
        return {"forecast_df": pd.DataFrame(), "metrics": {}, "model_name": "EnsembleWeighted", "available": False}

    joined = pd.concat(dfs, axis=1)

    # Determine weights
    weights = None
    if any(m is not None for m in model_mapes):
        inv = []
        for m in model_mapes:
            if m is None:
                inv.append(1.0)
            else:
                inv.append(1.0 / (float(m) + eps))
        arr = np.array(inv, dtype=float)
        weights = arr / arr.sum()
    else:
        weights = np.ones(len(dfs), dtype=float) / len(dfs)

    # apply weights (align columns) — if joined has NaNs, fill with column mean
    joined = joined.fillna(joined.mean())
    weighted = joined.multiply(weights, axis=1).sum(axis=1)
    forecast_df = weighted.reset_index().rename(columns={0: "Forecast"})

    # Aggregate metrics similarly to average but weighted when numeric
    metric_keys = set().union(*(set(f.get("metrics", {}).keys()) for f in forecasts))
    agg_metrics = {}
    for k in metric_keys:
        vals = []
        wvals = []
        for i, f in enumerate(forecasts):
            m = f.get("metrics", {})
            if k in m and isinstance(m[k], (int, float)):
                vals.append(float(m[k]))
                wvals.append(weights[i] if i < len(weights) else 0.0)
        if vals:
            if len(wvals) == len(vals):
                agg_metrics[k] = float(np.dot(np.array(vals), np.array(wvals)) / (np.sum(wvals) if np.sum(wvals) > 0 else 1.0))
            else:
                agg_metrics[k] = sum(vals) / len(vals)

    return {
        "forecast_df": forecast_df,
        "metrics": agg_metrics,
        "model_name": "EnsembleWeighted",
        "available": True,
    }


__all__ = ["ensemble_average"]
