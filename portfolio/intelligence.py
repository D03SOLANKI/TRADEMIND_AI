"""TradeMind AI — Portfolio Intelligence Engine (Phase 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List

from data.loader import future_dates
from features.indicators import add_all
from models import arima as arima_model
from models import garch as garch_model
from models import xgboost_model as xgb_model
from signals.backtester import run_backtest


@dataclass
class PortfolioHealth:
    score: float
    health_label: str
    risk_label: str
    components: Dict[str, float]
    suggestions: List[str]


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, v)))


def infer_sector(ticker: str) -> str:
    t = (ticker or "").upper()
    if t.endswith("-USD"):
        return "Crypto"
    if t.endswith("=X"):
        return "Forex"
    if t.endswith(".NS") or t.endswith(".BO"):
        return "India Equity"
    banks = {"JPM", "BAC", "WFC", "GS", "MS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"}
    tech = {"AAPL", "MSFT", "GOOGL", "GOOG", "NVDA", "META", "AMZN", "TSLA", "INFY.NS", "TCS.NS"}
    energy = {"XOM", "CVX", "BP", "SHEL", "RELIANCE.NS", "ONGC.NS"}
    pharma = {"JNJ", "PFE", "MRK", "SUNPHARMA.NS", "DRREDDY.NS"}
    if t in banks:
        return "Financials"
    if t in tech:
        return "Technology"
    if t in energy:
        return "Energy"
    if t in pharma:
        return "Healthcare"
    return "Other"


def portfolio_positions_frame(portfolio, live_prices: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for ticker, pos in portfolio.positions.items():
        px = float(live_prices.get(ticker) or 0.0)
        if px <= 0:
            continue
        value = pos.value(px)
        rows.append(
            {
                "Ticker": ticker,
                "Sector": infer_sector(ticker),
                "Shares": float(pos.shares),
                "Avg Price": float(pos.avg_price),
                "Live Price": px,
                "Invested": float(pos.cost_basis),
                "Value": float(value),
                "PnL": float(pos.pnl(px)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    tv = out["Value"].sum()
    out["Weight"] = out["Value"] / tv if tv > 0 else 0.0
    out["Concentration"] = out["Weight"]
    return out.sort_values("Weight", ascending=False).reset_index(drop=True)


def _weighted_portfolio_returns(price_history: Dict[str, pd.DataFrame], weights: Dict[str, float]) -> pd.Series:
    if not price_history:
        return pd.Series(dtype=float)
    prices = {}
    for t, df in price_history.items():
        if df is None or df.empty or "Date" not in df.columns or "Close" not in df.columns:
            continue
        prices[t] = df.set_index("Date")["Close"].astype(float)
    if not prices:
        return pd.Series(dtype=float)
    price_df = pd.DataFrame(prices).dropna(how="all").ffill().dropna()
    if price_df.empty:
        return pd.Series(dtype=float)
    ret = price_df.pct_change().dropna()
    if ret.empty:
        return pd.Series(dtype=float)
    w = pd.Series(weights).reindex(ret.columns).fillna(0.0)
    if w.sum() == 0:
        return pd.Series(dtype=float)
    w = w / w.sum()
    return (ret * w).sum(axis=1)


def _risk_metrics(rets: pd.Series) -> Dict[str, float]:
    if rets.empty or len(rets) < 10:
        return {
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "VaR95": np.nan,
            "Max Drawdown": np.nan,
            "Ann Vol": np.nan,
        }
    mu = float(rets.mean())
    sig = float(rets.std())
    neg = rets[rets < 0]
    sig_d = float(neg.std()) if len(neg) > 1 else np.nan
    sharpe = (mu / sig) * np.sqrt(252) if sig > 0 else np.nan
    sortino = (mu / sig_d) * np.sqrt(252) if sig_d and sig_d > 0 else np.nan
    var95 = float(-np.percentile(rets, 5))
    ann_vol = sig * np.sqrt(252)

    eq = (1 + rets).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    mdd = float(dd.min()) if len(dd) else np.nan

    return {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "VaR95": var95,
        "Max Drawdown": mdd,
        "Ann Vol": ann_vol,
    }


def _health_score(weights: pd.Series, sector_w: pd.Series, risk: Dict[str, float], corr_pairs: List[tuple]) -> PortfolioHealth:
    n = len(weights)
    max_w = float(weights.max()) if n else 1.0
    hhi = float((weights ** 2).sum()) if n else 1.0

    if n > 1:
        diversification = _clip(((1 - hhi) / (1 - 1 / n)) * 100)
    else:
        diversification = 0.0

    concentration = _clip(100 - max_w * 120)
    ann_vol = float(risk.get("Ann Vol", np.nan))
    sharpe = float(risk.get("Sharpe", np.nan))
    vol_score = _clip(100 - (0 if np.isnan(ann_vol) else ann_vol * 260))
    sharpe_score = _clip(0 if np.isnan(sharpe) else ((sharpe + 1.0) / 3.0) * 100)
    sector_score = _clip(100 - (float(sector_w.max()) * 120 if len(sector_w) else 100))

    score = float(np.average(
        [diversification, concentration, vol_score, sharpe_score, sector_score],
        weights=[0.24, 0.2, 0.2, 0.22, 0.14],
    ))

    if score >= 75:
        health_label = "Excellent"
    elif score >= 60:
        health_label = "Good"
    elif score >= 45:
        health_label = "Moderate"
    else:
        health_label = "Fragile"

    if np.isnan(ann_vol):
        risk_label = "Unknown"
    elif ann_vol >= 0.35:
        risk_label = "High Risk"
    elif ann_vol >= 0.2:
        risk_label = "Moderate Risk"
    else:
        risk_label = "Low Risk"

    suggestions = []
    if max_w > 0.35:
        suggestions.append("Reduce single-asset concentration below 35%.")
    if len(corr_pairs) >= 2:
        suggestions.append("Trim highly correlated holdings to improve diversification.")
    if not np.isnan(ann_vol) and ann_vol > 0.3:
        suggestions.append("Portfolio volatility is elevated; consider defensive allocation.")
    if not np.isnan(sharpe) and sharpe < 0.75:
        suggestions.append("Sharpe ratio is weak; prioritize assets with stronger risk-adjusted return.")
    if len(sector_w) and float(sector_w.max()) > 0.45:
        suggestions.append("Sector exposure is imbalanced; spread capital across additional sectors.")
    if not suggestions:
        suggestions.append("Portfolio composition looks healthy; rebalance monthly to maintain weights.")

    return PortfolioHealth(
        score=round(score, 2),
        health_label=health_label,
        risk_label=risk_label,
        components={
            "Diversification": round(diversification, 2),
            "Concentration": round(concentration, 2),
            "Volatility": round(vol_score, 2),
            "Sharpe": round(sharpe_score, 2),
            "Sector": round(sector_score, 2),
        },
        suggestions=suggestions,
    )


def _per_asset_intelligence(price_history: Dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    rows = []
    for ticker, df in price_history.items():
        if df is None or df.empty or len(df) < 80:
            continue
        close_now = float(df["Close"].iloc[-1])
        expected = np.nan
        vol = np.nan
        prob_up = np.nan
        backtest_ret = np.nan
        used = []

        try:
            ar = arima_model.forecast(df, horizon)
            f_end = float(ar["forecast_df"]["Forecast"].iloc[-1])
            expected = (f_end / close_now - 1) * 100 if close_now else np.nan
            used.append("ARIMA")
        except Exception:
            pass

        try:
            gx = garch_model.forecast(df, max(7, min(horizon, 30)))
            vol = float(gx.get("vol_forecast", {}).get("30d", np.nan)) * 100
            used.append("GARCH")
        except Exception:
            pass

        try:
            xg = xgb_model.forecast(df, max(7, min(horizon, 30)))
            prob_up = float(xg.get("probability", np.nan)) * 100
            used.append("XGBoost")
        except Exception:
            pass

        try:
            feat = add_all(df)
            bt = run_backtest(feat, "RSI + MACD Combined", capital=10000.0)
            backtest_ret = float(bt["metrics"].get("Total Return %", np.nan))
            used.append("Backtest")
        except Exception:
            pass

        rows.append(
            {
                "Ticker": ticker,
                "Expected Return %": expected,
                "Vol Forecast %": vol,
                "Up Probability %": prob_up,
                "Backtest Return %": backtest_ret,
                "Model Blend": ", ".join(used) if used else "None",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out


def _portfolio_projection(
    total_value: float,
    horizon: int,
    expected_return_pct: float,
    ann_vol: float,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    horizon = int(max(7, horizon))
    if total_value <= 0:
        return pd.DataFrame()

    daily_ret = (1 + expected_return_pct / 100.0) ** (1 / horizon) - 1
    sigma_day = 0.0 if np.isnan(ann_vol) else float(ann_vol) / np.sqrt(252)

    dates = future_dates(reference_df, horizon)
    rows = []
    for i in range(1, horizon + 1):
        base = total_value * ((1 + daily_ret) ** i)
        band = 1.28 * sigma_day * np.sqrt(i)
        rows.append(
            {
                "Date": dates[i - 1],
                "Base": base,
                "Best": base * (1 + band),
                "Worst": base * max(0.0, 1 - band),
            }
        )
    return pd.DataFrame(rows)


def simulate_sip(current_value: float, monthly_contrib: float, years: int, annual_return: float) -> pd.DataFrame:
    months = int(max(1, years * 12))
    mret = (1 + annual_return) ** (1 / 12) - 1
    value = float(current_value)
    rows = []
    for m in range(1, months + 1):
        value = value * (1 + mret) + monthly_contrib
        rows.append({"Month": m, "Portfolio Value": value})
    return pd.DataFrame(rows)


def simulate_market_drop(positions_df: pd.DataFrame, drop_pct: float) -> pd.DataFrame:
    if positions_df.empty:
        return pd.DataFrame()
    out = positions_df[["Ticker", "Value", "Weight", "Sector"]].copy()
    out["Stress Value"] = out["Value"] * (1 - drop_pct / 100.0)
    out["Impact"] = out["Stress Value"] - out["Value"]
    return out.sort_values("Impact")


def simulate_asset_impact(positions_df: pd.DataFrame, ticker: str, change_value: float) -> pd.DataFrame:
    if positions_df.empty:
        return pd.DataFrame()
    out = positions_df[["Ticker", "Value"]].copy()
    if ticker in out["Ticker"].values:
        out.loc[out["Ticker"] == ticker, "Value"] += change_value
    else:
        out = pd.concat([out, pd.DataFrame([{"Ticker": ticker, "Value": change_value}])], ignore_index=True)
    out["Value"] = out["Value"].clip(lower=0.0)
    tv = out["Value"].sum()
    out["New Weight"] = out["Value"] / tv if tv > 0 else 0.0
    return out.sort_values("New Weight", ascending=False).reset_index(drop=True)


def analyze_portfolio(
    portfolio,
    live_prices: Dict[str, float],
    price_history: Dict[str, pd.DataFrame],
    horizon: int = 30,
) -> Dict[str, Any]:
    positions_df = portfolio_positions_frame(portfolio, live_prices)
    if positions_df.empty:
        return {
            "positions": positions_df,
            "allocation": pd.DataFrame(),
            "sectors": pd.DataFrame(),
            "correlation": pd.DataFrame(),
            "correlated_pairs": [],
            "risk": {},
            "health": None,
            "asset_intel": pd.DataFrame(),
            "projection": pd.DataFrame(),
            "rebalancing": pd.DataFrame(),
        }

    alloc = positions_df[["Ticker", "Value", "Weight"]].copy()
    sectors = positions_df.groupby("Sector", as_index=False)["Weight"].sum().sort_values("Weight", ascending=False)

    w = dict(zip(positions_df["Ticker"], positions_df["Weight"]))
    port_rets = _weighted_portfolio_returns(price_history, w)
    risk = _risk_metrics(port_rets)

    corr = pd.DataFrame()
    corr_pairs: List[tuple] = []
    if price_history:
        closes = {
            t: df.set_index("Date")["Close"].astype(float)
            for t, df in price_history.items()
            if df is not None and not df.empty and "Date" in df.columns and "Close" in df.columns
        }
        if closes:
            ret_df = pd.DataFrame(closes).dropna(how="all").ffill().dropna().pct_change().dropna()
            if not ret_df.empty:
                corr = ret_df.corr()
                cols = list(corr.columns)
                for i in range(len(cols)):
                    for j in range(i + 1, len(cols)):
                        c = float(corr.iloc[i, j])
                        if c >= 0.75:
                            corr_pairs.append((cols[i], cols[j], c))

    health = _health_score(positions_df["Weight"], sectors["Weight"] if not sectors.empty else pd.Series(dtype=float), risk, corr_pairs)

    asset_intel = _per_asset_intelligence(price_history, horizon)
    total_value = float(positions_df["Value"].sum())
    expected_port_return = np.nan
    if not asset_intel.empty:
        mix = asset_intel.merge(positions_df[["Ticker", "Weight"]], on="Ticker", how="left")
        mix["Expected Return %"] = mix["Expected Return %"].fillna(0.0)
        expected_port_return = float((mix["Expected Return %"] * mix["Weight"]).sum())

    ref_df = next((d for d in price_history.values() if d is not None and not d.empty), None)
    projection = _portfolio_projection(
        total_value=total_value,
        horizon=horizon,
        expected_return_pct=0.0 if np.isnan(expected_port_return) else expected_port_return,
        ann_vol=float(risk.get("Ann Vol", np.nan)),
        reference_df=ref_df if ref_df is not None else pd.DataFrame({"Date": [pd.Timestamp.today()]})
    )

    target_weight = 1.0 / len(positions_df)
    rebalancing = positions_df[["Ticker", "Weight", "Value"]].copy()
    rebalancing["Target Weight"] = target_weight
    rebalancing["Delta Weight"] = rebalancing["Target Weight"] - rebalancing["Weight"]
    rebalancing["Action"] = np.where(
        rebalancing["Delta Weight"] > 0.03,
        "Increase",
        np.where(rebalancing["Delta Weight"] < -0.03, "Trim", "Hold"),
    )

    return {
        "positions": positions_df,
        "allocation": alloc,
        "sectors": sectors,
        "correlation": corr,
        "correlated_pairs": sorted(corr_pairs, key=lambda x: x[2], reverse=True),
        "risk": risk,
        "health": health,
        "asset_intel": asset_intel,
        "projection": projection,
        "rebalancing": rebalancing.sort_values("Delta Weight"),
        "portfolio_expected_return_pct": expected_port_return,
        "portfolio_value": total_value,
    }
