"""TradeMind AI — Risk Metrics"""
import numpy as np
import pandas as pd
from config.settings import TRADING_DAYS, RISK_FREE


def compute(df: pd.DataFrame, garch: dict = None) -> dict:
    close = df["Close"].dropna()
    rets  = close.pct_change().dropna()
    if len(rets) < 10:
        return {}

    mu   = float(rets.mean())
    sig  = float(rets.std())
    rfr  = RISK_FREE / TRADING_DAYS
    neg  = rets[rets < 0]
    sigD = float(neg.std()) if len(neg) > 1 else np.nan

    sharpe  = np.sqrt(TRADING_DAYS) * (mu - rfr) / sig if sig else np.nan
    sortino = np.sqrt(TRADING_DAYS) * (mu - rfr) / sigD if sigD and sigD > 0 else np.nan

    cum   = (1 + rets).cumprod()
    peak  = cum.cummax()
    dd    = (cum - peak) / peak
    maxDD = float(dd.min())

    days  = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
    years = days / 365 if days > 0 else np.nan
    cagr  = (float(close.iloc[-1]) / float(close.iloc[0])) ** (1/years) - 1 if years else np.nan
    calmar= cagr / abs(maxDD) if maxDD != 0 else np.nan

    var95 = float(-np.percentile(rets, 5))
    annV  = sig * np.sqrt(TRADING_DAYS)

    metrics = {
        "Sharpe Ratio":      round(sharpe,  4) if not np.isnan(sharpe)  else None,
        "Sortino Ratio":     round(sortino, 4) if not np.isnan(sortino) else None,
        "Max Drawdown (%)":  round(maxDD*100, 2),
        "CAGR (%)":          round(cagr*100,  2) if not np.isnan(cagr)  else None,
        "Calmar Ratio":      round(calmar, 4)    if not np.isnan(calmar) else None,
        "VaR 95% (daily %)": round(var95*100, 3),
        "Ann. Volatility (%)": round(annV*100, 2),
    }

    if isinstance(garch, dict):
        vf = garch.get("vol_forecast", {})
        regime = garch.get("regime")
        clustering = None
        if len(rets) > 10:
            clustering = float(rets.pow(2).autocorr(lag=1))
        metrics.update({
            "GARCH 7d Vol (%)": round(vf.get("7d", np.nan) * 100, 2) if vf else None,
            "GARCH 14d Vol (%)": round(vf.get("14d", np.nan) * 100, 2) if vf else None,
            "GARCH 30d Vol (%)": round(vf.get("30d", np.nan) * 100, 2) if vf else None,
            "Vol Clustering": round(clustering, 3) if clustering is not None and not np.isnan(clustering) else None,
            "Stress Regime": "High" if regime == "High volatility" else "Moderate" if regime == "Moderate volatility" else "Low",
            "GARCH Regime": regime,
        })

    return metrics


def sharpe_label(v) -> str:
    if v is None or np.isnan(v): return "N/A"
    if v >= 2:   return "🟢 Excellent (>2)"
    if v >= 1:   return "🟡 Good (1-2)"
    if v >= 0.5: return "🟠 Acceptable (0.5-1)"
    return "🔴 Poor (<0.5)"
