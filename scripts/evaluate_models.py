"""Script to run and evaluate available models for a given ticker.

Usage (from project root):
  python -m scripts.evaluate_models --ticker AAPL --horizon 7

The script will attempt to run XGBoost, LSTM, ARIMA, Prophet (if installed),
collect metrics, and write a JSON summary to `tmp/test_results.json`.
"""
import argparse
import json
import traceback
from pathlib import Path

import pandas as pd

from data.loader import load_asset

from models import xgboost_model, arima, prophet, lstm, ensemble


def run_all(ticker: str, horizon: int, period: str = "1y", interval: str = "1d") -> dict:
    df = load_asset(ticker, period=period, interval=interval)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")

    results = {}
    models = [
        ("XGBoost", xgboost_model),
        ("LSTM", lstm),
        ("ARIMA", arima),
        ("Prophet", prophet),
    ]

    forecasts = []
    for name, mod in models:
        try:
            fc = mod.forecast(df, horizon)
            forecasts.append(fc)
            results[name] = {"ok": True, "metrics": fc.get("metrics", {}), "model_name": fc.get("model_name")}
        except Exception as e:
            results[name] = {"ok": False, "error": str(e)}
            results[name]["traceback"] = traceback.format_exc()

    # Ensemble
    try:
        ens = ensemble.ensemble_average(forecasts)
        results["Ensemble"] = {"ok": True, "metrics": ens.get("metrics", {})}
        # Weighted ensemble that prefers models with lower MAPE
        wens = ensemble.ensemble_weighted_by_mape(forecasts)
        results["EnsembleWeighted"] = {"ok": True, "metrics": wens.get("metrics", {})}
    except Exception as e:
        results["Ensemble"] = {"ok": False, "error": str(e)}

    # Save to tmp
    outdir = Path("tmp")
    outdir.mkdir(exist_ok=True)
    outfile = outdir / "test_results.json"
    with open(outfile, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--horizon", type=int, default=7)
    p.add_argument("--period", default="1y")
    p.add_argument("--interval", default="1d")
    args = p.parse_args()

    res = run_all(args.ticker, args.horizon, period=args.period, interval=args.interval)
    print("Evaluation results written to tmp/test_results.json")
    print(res)


if __name__ == "__main__":
    main()
