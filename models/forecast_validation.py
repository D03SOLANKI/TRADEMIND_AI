"""
TradeMind AI — Forecast Validation Backtesting
===============================================
Implements walk-forward / rolling forecast validation and forecast-driven strategy simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from models import arima as arima_model
from models import prophet as prophet_model
from models import lstm as lstm_model
from models import xgboost_model as xgboost_model
from config.settings import ENABLE_LSTM, ENABLE_PROPHET
from utils.evaluation import compute_rmse_mae, compute_metrics_normalized

MODEL_MAP = {
    "ARIMA": arima_model,
    "Prophet": prophet_model,
    "LSTM": lstm_model,
    "XGBoost": xgboost_model,
}

DEFAULT_MIN_TRAIN = 60
DEFAULT_ROLLING_WINDOW = 180
DEFAULT_HIT_THRESHOLD = 5.0  # percent


def _safe_model_forecast(model_name: str, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}")
    if model_name == "Prophet" and not ENABLE_PROPHET:
        raise ImportError("Prophet is not enabled or installed.")
    if model_name == "LSTM" and not ENABLE_LSTM:
        raise ImportError("LSTM is not enabled or installed.")

    model = MODEL_MAP[model_name]
    result = model.forecast(df, horizon)
    if "forecast_df" not in result:
        raise RuntimeError(f"{model_name} forecast did not return forecast_df")
    return result["forecast_df"]


def _safe_model_result(model_name: str, df: pd.DataFrame, horizon: int) -> Dict[str, Any]:
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}")
    if model_name == "Prophet" and not ENABLE_PROPHET:
        raise ImportError("Prophet is not enabled or installed.")
    if model_name == "LSTM" and not ENABLE_LSTM:
        raise ImportError("LSTM is not enabled or installed.")
    return MODEL_MAP[model_name].forecast(df, horizon)


def _sign(value: float, threshold_pct: float = 0.0) -> int:
    if abs(value) <= threshold_pct:
        return 0
    return 1 if value > 0 else -1


def _batch_metrics(base: float, actual: np.ndarray, predicted: np.ndarray, direction_threshold_pct: float = 0.0) -> Dict[str, Any]:
    actual = np.asarray(actual, dtype=float).flatten()
    predicted = np.asarray(predicted, dtype=float).flatten()
    n = min(len(actual), len(predicted))
    if n == 0:
        return {
            "rmse": 0.0,
            "mae": 0.0,
            "mape": 0.0,
            "bias_pct": 0.0,
            "avg_error_pct": 0.0,
            "direction_correct": 0,
            "direction_actual": 0,
            "direction_pred": 0,
        }

    actual = actual[:n]
    predicted = predicted[:n]

    rmse, mae = compute_rmse_mae(actual, predicted)
    mask = actual != 0
    mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100) if mask.any() else 0.0
    bias_pct = float(np.mean((predicted - actual) / actual) * 100) if mask.any() else 0.0

    actual_end = actual[-1]
    pred_end = predicted[-1]
    actual_return_pct = (actual_end / base - 1) * 100 if base else 0.0
    pred_return_pct = (pred_end / base - 1) * 100 if base else 0.0
    direction_actual = _sign(actual_return_pct, threshold_pct=direction_threshold_pct)
    direction_pred = _sign(pred_return_pct, threshold_pct=direction_threshold_pct)
    direction_correct = 1 if direction_actual == direction_pred else 0

    return {
        "rmse": round(rmse, 6),
        "mae": round(mae, 6),
        "mape": round(mape, 2),
        "bias_pct": round(bias_pct, 2),
        "avg_error_pct": round(float(np.mean(np.abs((predicted - actual) / np.where(actual != 0, actual, 1))) * 100), 2),
        "direction_correct": direction_correct,
        "direction_actual": direction_actual,
        "direction_pred": direction_pred,
    }


def walk_forward_validate(
    df: pd.DataFrame,
    model_name: str,
    horizon: int,
    method: str = "expanding",
    step: int = 7,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    min_train_size: int = DEFAULT_MIN_TRAIN,
    hit_threshold: float = DEFAULT_HIT_THRESHOLD,
    direction_threshold_pct: float = 2.0,
) -> Dict[str, Any]:
    if method not in {"expanding", "rolling"}:
        raise ValueError("method must be 'expanding' or 'rolling'")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    records: List[Dict[str, Any]] = []
    total_rows = len(df)
    if total_rows < min_train_size + horizon:
        raise ValueError(f"Need at least {min_train_size + horizon} rows for walk-forward validation.")

    last_index = total_rows - horizon
    for end in range(min_train_size, last_index + 1, step):
        train_start = 0 if method == "expanding" else max(0, end - rolling_window)
        train_df = df.iloc[train_start:end]
        if len(train_df) < min_train_size:
            continue

        result = _safe_model_result(model_name, train_df, horizon)
        forecast_df = result["forecast_df"]

        if model_name == "XGBoost":
            prob_up = float(result.get("probability", 0.5))
            predicted_dir = 1 if prob_up >= 0.5 else -1
            actual_slice = df["Close"].iloc[end : end + horizon].values
            if len(actual_slice) < horizon:
                break
            base_price = float(train_df["Close"].iloc[-1])
            actual_end = float(actual_slice[-1])
            actual_return_pct = (actual_end / base_price - 1) * 100 if base_price else 0.0
            actual_dir = _sign(actual_return_pct, threshold_pct=direction_threshold_pct)
            pred_return_pct = (prob_up - 0.5) * 8.0
            predicted_dir = _sign(pred_return_pct, threshold_pct=direction_threshold_pct)

            records.append({
                "Date": pd.Timestamp(df["Date"].iloc[end]),
                "Base": round(base_price, 4),
                "Actual": round(actual_end, 4),
                "Predicted": round(base_price * (1 + (prob_up - 0.5) * 0.04), 4),
                "Return Actual %": round(actual_return_pct, 2),
                "Return Predicted %": round(pred_return_pct, 2),
                "Error %": round(abs((1.0 if actual_dir > 0 else 0.0) - prob_up) * 100, 2),
                "Direction Correct": bool(actual_dir == predicted_dir),
                "Actual Direction": "Up" if actual_dir > 0 else "Down" if actual_dir < 0 else "Flat",
                "Predicted Direction": "Up" if predicted_dir > 0 else "Down" if predicted_dir < 0 else "Flat",
                "RMSE": np.nan,
                "MAE": np.nan,
                "MAPE": np.nan,
                "Bias %": round((prob_up - 0.5) * 100, 2),
                "Hit": 100.0 if actual_dir == predicted_dir else 0.0,
            })
            continue

        predicted = forecast_df["Forecast"].values
        actual_slice = df["Close"].iloc[end : end + horizon].values
        if len(actual_slice) < horizon:
            break

        base_price = float(train_df["Close"].iloc[-1])
        actual_end = float(actual_slice[-1])
        predicted_end = float(predicted[-1])
        predicted_return = (predicted_end / base_price - 1) * 100 if base_price else 0.0
        actual_return = (actual_end / base_price - 1) * 100 if base_price else 0.0
        error_pct = np.abs((predicted - actual_slice) / np.where(actual_slice != 0, actual_slice, 1)) * 100
        batch = _batch_metrics(base_price, actual_slice, predicted, direction_threshold_pct=direction_threshold_pct)

        records.append({
            "Date": pd.Timestamp(df["Date"].iloc[end]),
            "Base": round(base_price, 4),
            "Actual": round(actual_end, 4),
            "Predicted": round(predicted_end, 4),
            "Return Actual %": round(actual_return, 2),
            "Return Predicted %": round(predicted_return, 2),
            "Error %": round(float(np.mean(error_pct)), 2),
            "Direction Correct": bool(batch["direction_correct"]),
            "Actual Direction": "Up" if batch["direction_actual"] > 0 else "Down" if batch["direction_actual"] < 0 else "Flat",
            "Predicted Direction": "Up" if batch["direction_pred"] > 0 else "Down" if batch["direction_pred"] < 0 else "Flat",
            "RMSE": batch["rmse"],
            "MAE": batch["mae"],
            "MAPE": batch["mape"],
            "Bias %": batch["bias_pct"],
            "Hit": float(np.mean(error_pct <= hit_threshold)) * 100,
        })

    if not records:
        raise RuntimeError("Walk-forward validation produced no windows. Try smaller step or more data.")

    history = pd.DataFrame(records)
    summary = {
        "Model": model_name,
        "Direction Accuracy": round(history["Direction Correct"].mean() * 100, 2),
        "Forecast Hit Rate": round(history["Hit"].mean(), 2),
        "Avg Error %": round(history["Error %"].mean(), 2),
        "RMSE": round(history["RMSE"].mean(), 6),
        "MAE": round(history["MAE"].mean(), 6),
        "MAPE": round(history["MAPE"].mean(), 2),
        "Bias %": round(history["Bias %"].mean(), 2),
    }

    direction_pct = summary["Direction Accuracy"]
    mape = summary["MAPE"]
    hit_rate = summary["Forecast Hit Rate"]
    bias = abs(summary["Bias %"])
    
    # Component 1: Directional Accuracy (40% weight)
    # Primary signal - how well does model predict up/down
    direction_score = direction_pct
    
    # Component 2: Hit Rate / Forecast Quality (30% weight)
    # Secondary signal - what % of predictions fall within threshold
    hit_score = hit_rate
    
    # Component 3: Error Quality Score (20% weight)
    # Tertiary signal - combines MAPE penalization
    error_score = max(0.0, 100 - min(100.0, mape * 3.0))
    
    # Component 4: Bias Penalty (10% weight)
    # Reduce confidence if model has systematic bias
    # Bias > 10% is problematic, bias > 20% is critical
    if bias <= 5.0:
        bias_score = 100.0
    elif bias <= 10.0:
        bias_score = 95.0 - (bias - 5.0) * 2
    elif bias <= 20.0:
        bias_score = max(10.0, 85.0 - (bias - 10.0) * 3)
    else:
        bias_score = max(0.0, 55.0 - (bias - 20.0) * 1.5)
    
    # Component 5: Consistency Penalty (optional boost)
    # If standard deviation of errors is low, model is more consistent
    avg_error = summary["Avg Error %"]
    consistency_bonus = max(0.0, 5.0 - min(5.0, avg_error * 0.25))
    
    # Combined confidence with improved weighting
    confidence = int(max(0, min(100, 
        direction_score * 0.40 + 
        hit_score * 0.30 + 
        error_score * 0.20 + 
        bias_score * 0.10 + 
        consistency_bonus
    )))
    
    summary["Forecast Confidence"] = confidence
    summary["Up/Down Accuracy"] = summary["Direction Accuracy"]
    summary["Threshold Hit Accuracy"] = round(history["Hit"].mean(), 2)

    return {
        "history": history,
        "summary": summary,
        "model_name": model_name,
        "method": method,
        "horizon": horizon,
        "step": step,
        "rolling_window": rolling_window,
    }


def compare_models_accuracy(
    df: pd.DataFrame,
    horizon: int,
    method: str = "expanding",
    step: int = 7,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    min_train_size: int = DEFAULT_MIN_TRAIN,
    hit_threshold: float = DEFAULT_HIT_THRESHOLD,
    direction_threshold_pct: float = 2.0,
    precomputed_summaries: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    precomputed_summaries = precomputed_summaries or {}

    for name in ["ARIMA", "Prophet", "LSTM", "XGBoost"]:
        if name == "Prophet" and not ENABLE_PROPHET:
            continue
        if name == "LSTM" and not ENABLE_LSTM:
            continue
        try:
            if name in precomputed_summaries:
                summary = precomputed_summaries[name]
            else:
                result = walk_forward_validate(
                    df, name, horizon, method=method,
                    step=step, rolling_window=rolling_window,
                    min_train_size=min_train_size, hit_threshold=hit_threshold,
                    direction_threshold_pct=direction_threshold_pct,
                )
                summary = result["summary"]

            score = summary["Direction Accuracy"] * 0.4 + (100 - summary["MAPE"]) * 0.35 + summary["Forecast Confidence"] * 0.25
            rows.append({
                "Model": name,
                "Direction Accuracy": summary["Direction Accuracy"],
                "Forecast Hit Rate": summary["Forecast Hit Rate"],
                "MAPE": summary["MAPE"],
                "Bias %": summary["Bias %"],
                "Confidence": summary["Forecast Confidence"],
                "Score": round(score, 2),
            })
        except Exception as exc:
            errors[name] = str(exc)

    comparison = pd.DataFrame(rows).sort_values(["Score", "Direction Accuracy"], ascending=[False, False]).reset_index(drop=True)
    best_model = comparison.iloc[0]["Model"] if not comparison.empty else None
    return {
        "comparison": comparison,
        "best_model": best_model,
        "errors": errors,
    }


def simulate_forecast_strategy(
    history: pd.DataFrame,
    threshold_pct: float = 2.0,
    capital: float = 10000.0,
) -> Dict[str, Any]:
    if history.empty:
        return {}

    eq = [capital]
    positions = []
    wins = []
    losses = []
    trades = []

    for _, row in history.iterrows():
        pred = float(row["Return Predicted %"])
        actual = float(row["Return Actual %"])
        if pred > threshold_pct:
            pnl = actual / 100
            signal = "BUY"
        elif pred < -threshold_pct:
            pnl = -actual / 100
            signal = "SELL"
        else:
            pnl = 0.0
            signal = "HOLD"

        equity = eq[-1] * (1 + pnl)
        eq.append(equity)
        trades.append({
            "Date": row["Date"],
            "Signal": signal,
            "Predicted %": round(pred, 2),
            "Actual %": round(actual, 2),
            "P&L %": round(pnl * 100, 2),
        })
        if signal in {"BUY", "SELL"}:
            if pnl > 0:
                wins.append(pnl)
            else:
                losses.append(pnl)

    equity_curve = pd.DataFrame({
        "Step": range(len(eq)),
        "Equity": eq,
        "Date": [pd.Timestamp(history["Date"].iloc[0])] + list(history["Date"]),
    })

    returns = [t["P&L %"] for t in trades if t["Signal"] != "HOLD"]
    total_trades = len([t for t in trades if t["Signal"] != "HOLD"])
    win_rate = float(len([r for r in returns if r > 0]) / total_trades * 100) if total_trades else 0.0
    avg_return = float(np.mean(returns)) if returns else 0.0
    std_return = float(np.std(returns)) if returns else 0.0
    sharpe = (avg_return / std_return * np.sqrt(252)) if total_trades and std_return != 0 else 0.0

    peak = -np.inf
    drawdowns = []
    for v in eq:
        peak = max(peak, v)
        drawdowns.append((v - peak) / peak * 100 if peak else 0.0)
    max_dd = min(drawdowns) if drawdowns else 0.0

    profit_factor = abs(sum([r for r in returns if r > 0]) / sum([r for r in returns if r < 0])) if any(r < 0 for r in returns) else float("inf")

    return {
        "equity_curve": equity_curve,
        "trades": pd.DataFrame(trades),
        "metrics": {
            "Total Return %": round((eq[-1] / capital - 1) * 100, 2),
            "Win Rate %": round(win_rate, 2),
            "Sharpe": round(sharpe, 3),
            "Max Drawdown %": round(max_dd, 2),
            "Profit Factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
            "Total Trades": total_trades,
            "Average Trade %": round(avg_return, 2),
        },
    }
