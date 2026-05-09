"""Run hyperparameter tuning for XGBoost and LSTM on a given ticker.

Saves results to `tmp/tune_results.json`.
"""
import argparse
import json
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

from data.loader import load_asset
from models import xgboost_model
from utils import tuning
from utils.tuning import tune_prophet_simple


def tune_xgb_for_df(df: pd.DataFrame):
    X, y, valid = xgboost_model._prepare_dataset(df)
    res = tuning.tune_xgb_classifier(X, y, n_iter=20, cv=3)
    return {
        "best_params": {k: (str(v) if not isinstance(v, (int, float, bool)) else v) for k,v in res.get("best_params", {}).items()},
        "cv_best_score": float(res.get("best_estimator").score(X, y)) if res.get("best_estimator") is not None else None,
    }


def tune_lstm_for_df(df: pd.DataFrame):
    close = df["Close"].values.astype(float)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close.reshape(-1,1)).flatten()
    # pick lookback 10 or based on length
    lookback = min(10, max(1, len(scaled)//10))
    best = tuning.tune_lstm_simple(scaled, lookback=lookback, epochs=8)
    return {"best_config": best}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", required=True)
    p.add_argument("--period", default="1y")
    p.add_argument("--interval", default="1d")
    args = p.parse_args()

    out = {}
    try:
        df = load_asset(args.ticker, period=args.period, interval=args.interval)
        if df.empty:
            raise RuntimeError("No data")

        out["XGBoost"] = tune_xgb_for_df(df)
        out["LSTM"] = tune_lstm_for_df(df)
        try:
            out["Prophet"] = tune_prophet_simple(df)
        except Exception as e:
            out["Prophet"] = {"error": str(e)}
    except Exception as e:
        out["error"] = str(e)
        out["traceback"] = traceback.format_exc()

    outdir = Path("tmp")
    outdir.mkdir(exist_ok=True)
    with open(outdir / "tune_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)

    print("Tuning complete. Results written to tmp/tune_results.json")


if __name__ == "__main__":
    main()
