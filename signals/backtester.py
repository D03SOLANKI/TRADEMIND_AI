"""TradeMind AI — Backtesting Engine (5 strategies + Forecast-Based)"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from config.settings import COMMISSION, SLIPPAGE


@dataclass
class Trade:
    entry_date:  object
    exit_date:   object
    entry_price: float
    exit_price:  float
    pnl_pct:     float = 0.0
    reason_in:   str   = ""
    reason_out:  str   = ""

    def __post_init__(self):
        cost = (COMMISSION + SLIPPAGE) * 2
        self.pnl_pct = (self.exit_price / self.entry_price - 1) * 100 - cost * 100


def _rsi_signals(df):
    if "RSI" not in df.columns:
        return pd.Series(0, index=df.index)
    rsi = df["RSI"]
    s   = pd.Series(0, index=df.index)
    s[(rsi.shift(1) < 30) & (rsi >= 30)] =  1
    s[(rsi.shift(1) < 70) & (rsi >= 70)] = -1
    return s


def _macd_signals(df):
    if "MACD" not in df.columns:
        return pd.Series(0, index=df.index)
    m, sg = df["MACD"], df["MACD_sig"]
    s     = pd.Series(0, index=df.index)
    s[(m.shift(1) < sg.shift(1)) & (m >= sg)] =  1
    s[(m.shift(1) > sg.shift(1)) & (m <= sg)] = -1
    return s


def _bb_signals(df):
    if "BB_upper" not in df.columns:
        return pd.Series(0, index=df.index)
    c, u, l = df["Close"], df["BB_upper"], df["BB_lower"]
    s = pd.Series(0, index=df.index)
    s[c <= l] =  1
    s[c >= u] = -1
    return s


def _sma_signals(df):
    if "SMA7" not in df.columns:
        return pd.Series(0, index=df.index)
    s7, s30 = df["SMA7"], df["SMA30"]
    s = pd.Series(0, index=df.index)
    s[(s7.shift(1) <= s30.shift(1)) & (s7 > s30)] =  1
    s[(s7.shift(1) >= s30.shift(1)) & (s7 < s30)] = -1
    return s


def _combo_signals(df):
    return (((_rsi_signals(df) == 1) & (_macd_signals(df) == 1)).astype(int)
          - ((_rsi_signals(df) == -1) & (_macd_signals(df) == -1)).astype(int))


def _forecast_signals(df, threshold_pct: float = 0.5):
    """
    Generate signals based on forecast vs actual price.
    Buy if forecast > close + threshold, Sell if forecast < close - threshold
    """
    if "Forecast" not in df.columns:
        return pd.Series(0, index=df.index)
    
    forecast = df["Forecast"]
    close = df["Close"]
    s = pd.Series(0, index=df.index)
    
    # Generate buy/sell based on forecast price movement
    forecast_ret = (forecast - close) / close * 100  # Return % implied by forecast
    s[forecast_ret > threshold_pct] = 1   # Bullish
    s[forecast_ret < -threshold_pct] = -1  # Bearish
    
    return s


def _forecast_and_technical_signals(df, threshold_pct: float = 0.5):
    """
    Hybrid strategy: Forecast + Technical Confirmation
    Buy only when BOTH forecast and technical indicators agree
    This reduces false signals and improves win rate
    """
    forecast_sig = _forecast_signals(df, threshold_pct)
    rsi_sig = _rsi_signals(df)
    macd_sig = _macd_signals(df)
    
    s = pd.Series(0, index=df.index)
    # Strong buy: forecast bullish + (RSI oversold OR MACD bullish)
    s[((forecast_sig == 1) & (rsi_sig == 1)) | ((forecast_sig == 1) & (macd_sig == 1))] = 1
    # Strong sell: forecast bearish + (RSI overbought OR MACD bearish)
    s[((forecast_sig == -1) & (rsi_sig == -1)) | ((forecast_sig == -1) & (macd_sig == -1))] = -1
    
    return s


def _forecast_only_signals(df):
    """Wrapper for forecast signals with default threshold"""
    return _forecast_signals(df, threshold_pct=0.5)


def _forecast_hybrid_signals(df):
    """Wrapper for hybrid forecast+technical signals"""
    return _forecast_and_technical_signals(df, threshold_pct=0.5)


STRATEGIES = {
    "RSI (30/70)":              _rsi_signals,
    "MACD Crossover":           _macd_signals,
    "Bollinger Reversion":      _bb_signals,
    "SMA Golden Cross":         _sma_signals,
    "RSI + MACD Combined":      _combo_signals,
    "Forecast-Based":           _forecast_only_signals,
    "Forecast + Technical":     _forecast_hybrid_signals,
}
STRATEGY_NAMES = list(STRATEGIES.keys())


def run_backtest(df: pd.DataFrame, strategy: str, capital: float = 10_000) -> dict:
    sigs   = STRATEGIES[strategy](df)
    closes = df["Close"].values
    dates  = df["Date"].values

    trades: List[Trade] = []
    equity, pos, ep, ed = capital, None, 0.0, None

    for i in range(1, len(df)):
        px, dt, sig = float(closes[i]), dates[i], int(sigs.iloc[i])
        if sig == 1 and pos is None:
            pos, ep, ed = "LONG", px * (1+SLIPPAGE), dt
        elif sig == -1 and pos == "LONG":
            t = Trade(ed, dt, ep, px*(1-SLIPPAGE), reason_in=f"{strategy} BUY", reason_out=f"{strategy} SELL")
            trades.append(t); equity *= (1 + t.pnl_pct/100); pos = None

    if pos == "LONG":
        t = Trade(ed, dates[-1], ep, float(closes[-1])*(1-SLIPPAGE),
                  reason_in=f"{strategy} BUY", reason_out="Period end")
        trades.append(t); equity *= (1 + t.pnl_pct/100)

    pnls  = [t.pnl_pct for t in trades]
    wins  = [p for p in pnls if p > 0]
    loss  = [p for p in pnls if p <= 0]

    # Equity curve
    eq_curve = [capital]
    e = capital
    for p in pnls:
        e *= (1 + p/100); eq_curve.append(e)

    # Drawdown of strategy
    peak, mdd = capital, 0.0
    for v in eq_curve:
        peak = max(peak, v)
        mdd  = min(mdd, (v - peak)/peak*100)

    n_trades  = len(pnls)
    win_rate  = len(wins)/n_trades*100 if n_trades else 0
    pf        = abs(sum(wins)/sum(loss)) if loss and sum(loss) != 0 else float("inf")
    sharpe    = (np.mean(pnls)/np.std(pnls)*np.sqrt(n_trades/max((df["Date"].iloc[-1]-df["Date"].iloc[0]).days/365,0.1))
                 if n_trades>1 and np.std(pnls)>0 else 0.0)
    bh        = (float(closes[-1])/float(closes[0])-1)*100

    log = pd.DataFrame([{
        "Entry":       pd.Timestamp(t.entry_date).date(),
        "Exit":        pd.Timestamp(t.exit_date).date(),
        "Buy Price":   round(t.entry_price,4),
        "Sell Price":  round(t.exit_price,4),
        "P&L %":       round(t.pnl_pct,2),
        "Result":      "✅" if t.pnl_pct>0 else "❌",
    } for t in trades]) if trades else pd.DataFrame()

    # Full equity curve DataFrame aligned to df dates
    eq_dates = [pd.Timestamp(dates[0])]
    for t in trades:
        eq_dates.append(pd.Timestamp(t.exit_date))
    eq_df = pd.DataFrame({"Date": eq_dates, "Equity": eq_curve[:len(eq_dates)]})

    return {
        "trades":       trades,
        "trade_log":    log,
        "equity_curve": eq_df,
        "metrics": {
            "Total Trades":   n_trades,
            "Win Trades":     len(wins),
            "Loss Trades":    len(loss),
            "Win Rate %":     round(win_rate,1),
            "Total Return %": round((equity/capital-1)*100,2),
            "Max Drawdown %": round(mdd,2),
            "Sharpe Ratio":   round(sharpe,3),
            "Profit Factor":  round(pf,2) if pf!=float("inf") else 999,
            "Avg Win %":      round(np.mean(wins),2)  if wins else 0,
            "Avg Loss %":     round(np.mean(loss),2)  if loss else 0,
            "Final Equity":   round(equity,2),
        },
        "buy_hold_return": round(bh,2),
    }
