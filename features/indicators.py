"""TradeMind AI — Technical Indicators"""
import numpy as np
import pandas as pd
from config.settings import (
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIG,
    SMA_S, SMA_L, BB_PERIOD, BB_STD, ATR_PERIOD,
)


def add_rsi(df):
    d = df["Close"].diff()
    g = d.where(d > 0, 0.0).rolling(RSI_PERIOD).mean()
    l = (-d.where(d < 0, 0.0)).rolling(RSI_PERIOD).mean()
    df["RSI"] = 100 - (100 / (1 + g / l))
    return df


def add_macd(df):
    e12 = df["Close"].ewm(span=MACD_FAST, adjust=False).mean()
    e26 = df["Close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["MACD"] = e12 - e26
    df["MACD_sig"] = df["MACD"].ewm(span=MACD_SIG, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]
    return df


def add_sma(df):
    df[f"SMA{SMA_S}"]  = df["Close"].rolling(SMA_S).mean()
    df[f"SMA{SMA_L}"]  = df["Close"].rolling(SMA_L).mean()
    return df


def add_ema(df):
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    return df


def add_bb(df):
    ma  = df["Close"].rolling(BB_PERIOD).mean()
    sd  = df["Close"].rolling(BB_PERIOD).std()
    df["BB_mid"]   = ma
    df["BB_upper"] = ma + BB_STD * sd
    df["BB_lower"] = ma - BB_STD * sd
    df["BB_pct"]   = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / ma
    return df


def add_atr(df):
    if not {"High","Low","Close"}.issubset(df.columns):
        df["ATR"] = np.nan; return df
    h, l, pc = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(ATR_PERIOD).mean()
    return df


def add_vol(df):
    df["Ret"]   = df["Close"].pct_change()
    df["Vol7"]  = df["Ret"].rolling(7).std()
    df["Vol30"] = df["Ret"].rolling(30).std()
    return df


def add_stoch(df):
    if not {"High","Low","Close"}.issubset(df.columns):
        return df
    low14  = df["Low"].rolling(14).min()
    high14 = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low14) / (high14 - low14 + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    return df


def add_vwap(df):
    if "Volume" not in df.columns:
        return df
    tp = (df.get("High", df["Close"]) + df.get("Low", df["Close"]) + df["Close"]) / 3
    df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
    return df


def detect_patterns(df):
    if not {"Open","High","Low","Close"}.issubset(df.columns):
        df["Pattern"] = ""; return df
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = (c - o).abs(); rng = h - l
    ush  = h - c.clip(upper=o); lsh = o.clip(upper=c) - l
    pats = np.full(len(df), "", dtype=object)

    pats[(rng > 3*body) & (lsh > 2*body) & (ush < body)] = "🔨 Hammer"
    pats[(rng > 3*body) & (ush > 2*body) & (lsh < body)] = "⭐ Shooting Star"

    po, pc   = o.shift(1), c.shift(1)
    pb, cb   = (pc-po).abs(), (c-o).abs()
    bull_eng = (pc<po)&(c>o)&(cb>pb)&(o<=pc)&(c>=po)
    bear_eng = (pc>po)&(c<o)&(cb>pb)&(o>=pc)&(c<=po)
    pats[bull_eng] = "🟢 Bull Engulf"
    pats[bear_eng] = "🔴 Bear Engulf"
    df["Pattern"] = pats
    return df


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for fn in [add_rsi, add_macd, add_sma, add_ema, add_bb,
               add_atr, add_vol, add_stoch, add_vwap, detect_patterns]:
        try:
            df = fn(df)
        except Exception:
            pass
    return df


def indicator_table(df: pd.DataFrame) -> pd.DataFrame:
    last = df.dropna(subset=["Close"]).iloc[-1]
    g = lambda c: float(last.get(c, np.nan)) if c in last.index else np.nan
    rows = []

    rsi = g("RSI")
    if not np.isnan(rsi):
        sig = ("Overbought","🔴") if rsi>70 else ("Oversold","🟢") if rsi<30 else ("Neutral","🟡")
        rows.append({"Indicator":"RSI (14)", "Value":f"{rsi:.1f}", "Signal":f"{sig[0]} {sig[1]}"})

    macd, msig = g("MACD"), g("MACD_sig")
    if not np.isnan(macd):
        sig = ("Bullish","🟢") if macd>msig else ("Bearish","🔴")
        rows.append({"Indicator":"MACD", "Value":f"{macd:.4f}", "Signal":f"{sig[0]} {sig[1]}"})

    close, bbu, bbl = g("Close"), g("BB_upper"), g("BB_lower")
    if not np.isnan(bbu):
        sig = ("Above Band","🔴") if close>bbu else ("Below Band","🟢") if close<bbl else ("In Band","🟡")
        rows.append({"Indicator":"Bollinger", "Value":f"{close:.2f}", "Signal":f"{sig[0]} {sig[1]}"})

    s7, s30 = g(f"SMA{SMA_S}"), g(f"SMA{SMA_L}")
    if not np.isnan(s7) and not np.isnan(s30):
        sig = ("Golden Cross","🟢") if s7>s30 else ("Death Cross","🔴")
        rows.append({"Indicator":f"SMA {SMA_S}/{SMA_L}", "Value":f"{s7:.2f}/{s30:.2f}", "Signal":f"{sig[0]} {sig[1]}"})

    sk = g("Stoch_K")
    if not np.isnan(sk):
        sig = ("Overbought","🔴") if sk>80 else ("Oversold","🟢") if sk<20 else ("Neutral","🟡")
        rows.append({"Indicator":"Stochastic", "Value":f"{sk:.1f}", "Signal":f"{sig[0]} {sig[1]}"})

    atr = g("ATR")
    if not np.isnan(atr):
        rows.append({"Indicator":"ATR (14)", "Value":f"{atr:.4f}", "Signal":"Volatility Gauge 📊"})

    return pd.DataFrame(rows)
