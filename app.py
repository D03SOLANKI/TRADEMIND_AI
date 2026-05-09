"""
TradeMind AI v3 — Main Application
=====================================
Run: streamlit run app.py

Architecture:
  app.py          ← UI entry point (this file)
  config/         ← settings, asset lists
  data/           ← fetch + preprocess (date-safe)
  features/       ← RSI, MACD, BB, ATR, Stoch, VWAP, Patterns
  models/         ← ARIMA, Prophet, LSTM (RangeIndex — no date bugs)
  signals/        ← WHY engine, backtester, sentiment
  risk/           ← Sharpe, Sortino, VaR, CAGR
  portfolio/      ← multi-asset P&L tracker
  alerts/         ← console / Telegram
  utils/          ← normalized evaluation metrics
"""

import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from config.settings   import MARKETS, PERIODS, INTERVALS, currency
from data.loader       import load_asset, fetch_live_price, safe_last_date
from features.indicators import add_all, indicator_table
from signals.signal_engine import generate
from signals.sentiment import analyze, news_sentiment
from signals.backtester import run_backtest, STRATEGY_NAMES
from risk.metrics      import compute as risk_compute, sharpe_label
from portfolio.tracker import Portfolio, Position
from portfolio.intelligence import (
    analyze_portfolio,
    simulate_sip,
    simulate_market_drop,
    simulate_asset_impact,
)
from alerts.alert_manager import AlertManager
from models            import arima as _arima, prophet as _prophet, lstm as _lstm, garch as _garch, xgboost_model as _xgboost
from models            import rl_agent as _rl
from models.comparator import run_all_models
from models.forecast_validation import (
    walk_forward_validate,
    compare_models_accuracy,
    simulate_forecast_strategy,
)

# ═══════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="TradeMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════
# PREMIUM CSS  (inspired by Bloomberg Terminal + TradingView)
# ═══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4, h5, h6 { font-family: 'Space Grotesk', sans-serif !important; }
.stApp { background: #0b0e14; color: #ecedf6; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #10131a !important;
    border-right: none;
}
section[data-testid="stSidebar"] .block-container { padding: 0 16px 24px; }
section[data-testid="stSidebar"] label { color: #a9abb3 !important; font-size: .8rem !important; font-family: 'Inter', sans-serif; font-weight: 500 !important; }
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stMultiSelect > div > div {
    background: #161a21 !important; border: 1px solid rgba(69, 72, 79, 0.15) !important; border-radius: 8px !important; color: #ecedf6 !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(34, 38, 47, 0.4); backdrop-filter: blur(20px);
    border: none; border-top: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 16px 20px; position: relative; overflow: hidden;
}
[data-testid="stMetricLabel"] { color: #a9abb3 !important; font-size: .75rem !important; text-transform: uppercase; letter-spacing: .5px; }
[data-testid="stMetricValue"] { color: #ecedf6 !important; font-weight: 700 !important; font-family: 'Space Grotesk', sans-serif !important; }
[data-testid="stMetricDelta"] { font-size: .8rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent; border-bottom: none;
    padding: 0 4px; gap: 4px; overflow-x: auto;
}
.stTabs [data-baseweb="tab"] {
    color: #a9abb3; font-weight: 500; font-family: 'Inter', sans-serif; font-size: .85rem;
    padding: 10px 16px; border-radius: 8px 8px 0 0;
    white-space: nowrap; transition: background .2s, color .2s;
}
.stTabs [aria-selected="true"] {
    color: #ba9eff !important; background: #161a21 !important;
    border-bottom: 2px solid #ba9eff !important;
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ba9eff 0%, #8455ef 100%);
    color: #000000; border: none; border-radius: 8px; font-weight: 600;
    font-size: .85rem; padding: 9px 20px; letter-spacing: .3px;
    transition: opacity .2s, transform .1s;
}
.stButton > button[kind="primary"]:hover { opacity: .88; transform: translateY(-1px); }
.stButton > button:not([kind="primary"]) {
    background: #22262f; border: none;
    color: #0EA5E9; border-radius: 8px; font-size: .83rem;
}
.stButton > button:not([kind="primary"]):hover { background: #282c36; transform: translateY(-1px); box-shadow: 0 0 24px rgba(186,158,255,0.08); }

/* ── DataFrames ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
.stDataFrame [data-testid="stDataFrameResizable"] { border: 1px solid rgba(69, 72, 79, 0.15) !important; border-radius: 10px !important; }
iframe { border-radius: 10px; }

/* ── Expanders ── */
div[data-testid="stExpander"] { background: #10131a; border: 1px solid rgba(69, 72, 79, 0.15) !important; border-radius: 10px !important; }
div[data-testid="stExpander"] summary { color: #ecedf6 !important; font-family: 'Inter', sans-serif; }

/* ── Alerts ── */
.stSuccess, .stInfo, .stWarning, .stError { border-radius: 8px !important; }

/* ── Custom Components ── */

/* Header */
.tm-hero {
    background: rgba(34, 38, 47, 0.4); backdrop-filter: blur(20px);
    border: none; border-top: 1px solid rgba(255,255,255,0.1); border-radius: 16px;
    padding: 28px 36px; margin-bottom: 20px; position: relative; overflow: hidden;
}
.tm-hero::before {
    content: ''; position: absolute; top: -60px; right: -60px;
    width: 260px; height: 260px; border-radius: 50%;
    background: radial-gradient(circle, rgba(186,158,255,.07) 0%, transparent 70%);
}
.tm-hero::after {
    content: ''; position: absolute; bottom: -40px; left: 20%;
    width: 180px; height: 180px; border-radius: 50%;
    background: radial-gradient(circle, rgba(52,181,250,.05) 0%, transparent 70%);
}
.tm-hero h1 {
    font-size: 2.1rem; font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #ba9eff 0%, #8455ef 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 6px;
}
.tm-hero p { color: #a9abb3; font-size: .95rem; }

/* Signal Card */
.sig-wrap { border-radius: 14px; padding: 22px 18px; text-align: center; position: relative; overflow: hidden; border-top: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(20px); }
.sig-BUY  { background: rgba(155, 255, 206, 0.1); }
.sig-SELL { background: rgba(255, 110, 132, 0.1); }
.sig-HOLD { background: rgba(245, 158, 11, 0.1); }
.sig-glow-BUY  { box-shadow: 0 0 24px rgba(155, 255, 206, 0.08); }
.sig-glow-SELL { box-shadow: 0 0 24px rgba(255, 110, 132, 0.08); }
.sig-glow-HOLD { box-shadow: 0 0 24px rgba(245, 158, 11, 0.08); }
.sig-label { font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 700; line-height: 1; }
.sig-BUY  .sig-label { color: #9bffce; }
.sig-SELL .sig-label { color: #ff6e84; }
.sig-HOLD .sig-label { color: #f59e0b; }
.sig-sub { font-size: .88rem; color: #a9abb3; font-family: 'Inter', sans-serif; margin-top: 8px; }

/* Score ring (Glow-Gauge) */
.score-row { display: flex; gap: 12px; justify-content: center; margin: 14px 0; }
.score-ring {
    width: 68px; height: 68px; border-radius: 50%;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    border: none; position: relative; flex-shrink: 0;
    background: rgba(34, 38, 47, 0.4); backdrop-filter: blur(5px);
}
.score-ring::before {
    content: ''; position: absolute; top:0; left:0; right:0; bottom:0; padding: 2px;
    border-radius: 50%; background: linear-gradient(180deg, inherit 0%, transparent 100%);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: destination-out; mask-composite: exclude;
}
.score-ring .num { font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; line-height: 1; color: #ecedf6; }
.score-ring .lbl { font-family: 'Inter', sans-serif; font-size: .6rem; letter-spacing: .4px; text-transform: uppercase; color: #a9abb3; margin-top: 2px; }
.sr-green::before { background: linear-gradient(180deg, #9bffce 0%, transparent 100%); }
.sr-yellow::before{ background: linear-gradient(180deg, #f59e0b 0%, transparent 100%); }
.sr-red::before   { background: linear-gradient(180deg, #ff6e84 0%, transparent 100%); }
.sr-green .num { color: #9bffce; }
.sr-yellow .num { color: #f59e0b; }
.sr-red .num { color: #ff6e84; }

/* Confidence bar */
.cbar-bg   { background: #161a21; border-radius: 99px; height: 6px; margin: 8px 0 2px; overflow: hidden; }
.cbar-fill { height: 6px; border-radius: 99px; transition: width .5s ease; box-shadow: 0 0 4px inherit; }

/* WHY items */
.why-item {
    display: flex; gap: 10px; align-items: flex-start;
    background: transparent; padding: 8px 0; margin: 3px 0;
}
.why-dot { width: 7px; height: 7px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; box-shadow: 0 0 6px inherit; }
.why-txt { font-size: .83rem; color: #ecedf6; font-family: 'Inter', sans-serif; line-height: 1.5; }
.risk-item {
    display: flex; gap: 10px; align-items: flex-start;
    background: transparent; padding: 8px 0; margin: 3px 0;
}
.risk-txt { font-size: .83rem; color: #ffb2b9; font-family: 'Inter', sans-serif; line-height: 1.5; }

/* Opportunity grid */
.opp-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 12px; margin: 10px 0; }
.opp-card {
    background: rgba(34, 38, 47, 0.4); backdrop-filter: blur(20px);
    border: none; border-top: 1px solid rgba(255,255,255,0.1); border-radius: 10px;
    padding: 12px 14px; cursor: pointer; transition: transform .15s, background .2s;
    box-shadow: 0 0 24px rgba(139, 92, 246, 0.03);
}
.opp-card:hover { background: #282c36; transform: translateY(-2px); box-shadow: 0 0 24px rgba(139, 92, 246, 0.08); }
.opp-ticker { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem; color: #ecedf6; }
.opp-sig    { font-size: .78rem; font-weight: 600; margin-top: 4px; font-family: 'Inter', sans-serif; }
.opp-conf   { font-size: .7rem; color: #a9abb3; margin-top: 2px; }

/* Sidebar logo */
.sb-logo { text-align: center; padding: 20px 0 16px; }
.sb-logo-icon { font-size: 2.4rem; }
.sb-logo-name { color: #ecedf6; font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 700; margin: 4px 0 2px; letter-spacing: -0.5px; }
.sb-logo-ver  { color: #a9abb3; font-size: .75rem; }
.sb-divider   { border: none; border-top: 1px solid rgba(69, 72, 79, 0.15); margin: 0 0 14px; }

/* Landing feature grid */
.feat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 24px 0; }
.feat-card { background: rgba(34, 38, 47, 0.4); border: none; border-top: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 20px 16px; backdrop-filter: blur(20px); }
.feat-icon { font-size: 1.8rem; margin-bottom: 10px; }
.feat-title { color: #ecedf6; font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1rem; margin-bottom: 4px; }
.feat-desc  { color: #a9abb3; font-size: .8rem; line-height: 1.5; }

/* Section separator */
.sec-div { border: none; height: 16px; background: transparent; margin: 20px 0; }

/* Metric pill */
.mpill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 99px; font-size: .75rem; font-weight: 600; margin: 2px;
}
.mpill-green { background: rgba(155, 255, 206, 0.1); color: #9bffce; border: none; }
.mpill-red   { background: rgba(255, 110, 132, 0.1); color: #ff6e84; border: none; }
.mpill-blue  { background: rgba(52, 181, 250, 0.1); color: #34b5fa; border: none; }
.mpill-gray  { background: #161a21; color: #a9abb3; border: none; }

h2, h3 { color: #ecedf6; font-family: 'Space Grotesk', sans-serif !important; }
p { color: #a9abb3; font-family: 'Inter', sans-serif; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0b0e14; }
::-webkit-scrollbar-thumb { background: #22262f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# CHART THEME
# ═══════════════════════════════════════════════════════
CT = dict(
    plot_bgcolor="#0b0e14", paper_bgcolor="#0b0e14",
    font=dict(color="#a9abb3", size=11, family="Inter"),
    xaxis=dict(showgrid=True, gridcolor="rgba(69, 72, 79, 0.15)", zeroline=False,
               showline=False, tickfont=dict(size=10)),
    yaxis=dict(showgrid=True, gridcolor="rgba(69, 72, 79, 0.15)", zeroline=False,
               showline=False, tickfont=dict(size=10)),
    legend=dict(bgcolor="rgba(22, 26, 33, 0.8)", bordercolor="rgba(69, 72, 79, 0.15)", borderwidth=1,
                font=dict(size=11, color="#ecedf6")),
    margin=dict(l=8, r=8, t=36, b=8),
    hovermode="x unified",
    hoverlabel=dict(bgcolor="#161a21", bordercolor="rgba(69, 72, 79, 0.3)",
                    font=dict(color="#ecedf6", size=11)),
)

# ═══════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════
_defaults = {
    "data":          {},
    "portfolio":     Portfolio(),
    "alerts":        AlertManager(),
    "last_forecast": {},
    "last_garch":    {},
    "last_ml":       {},
    "last_rl":       {},
    "last_sentiment": {"polarity": 0.0, "subjectivity": 0.0},
    "portfolio_ai_cache": {},
    "rl_lab_cache":  {},
    "price_alerts":  [],
    "watchlist":     [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════
def _bar(pct, color):
    st.markdown(
        f'<div class="cbar-bg"><div class="cbar-fill" '
        f'style="width:{pct}%;background:{color}"></div></div>',
        unsafe_allow_html=True,
    )

def _sr_cls(v):
    return "sr-green" if v >= 65 else "sr-yellow" if v >= 40 else "sr-red"

DEFAULT_ROLLING_WINDOW = 180

def _candle(df, title="", tail=None):
    d = df.tail(tail) if tail else df
    if all(c in d.columns for c in ["Open","High","Low"]):
        fig = go.Figure(go.Candlestick(
            x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
            increasing=dict(line=dict(color="#10B981", width=1.5), fillcolor="rgba(16, 185, 129, 0.15)"),
            decreasing=dict(line=dict(color="#ff6e84", width=1.5), fillcolor="rgba(255, 110, 132, 0.15)"),
        ))
        fig.update_layout(xaxis_rangeslider_visible=False, **CT)
    else:
        fig = go.Figure(go.Scatter(x=d["Date"], y=d["Close"],
                                   line=dict(color="#0EA5E9", width=2), name="Close"))
        fig.update_layout(**CT)
    fig.update_layout(title=dict(text=title, font=dict(size=13, color="#ecedf6", family="Space Grotesk")))
    return fig

def _forecast_chart(df, fdf, model, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"],
                             name="Historical", line=dict(color="#0EA5E9", width=2)))
    fig.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Forecast"],
                             name=f"{model} Forecast",
                             line=dict(color="#f59e0b", width=2, dash="dash"),
                             mode="lines+markers",
                             marker=dict(size=4, color="#f59e0b")))
    # Confidence band (simple ±2% around forecast)
    fc = fdf["Forecast"].values
    fig.add_trace(go.Scatter(
        x=pd.concat([fdf["Date"], fdf["Date"][::-1]]),
        y=np.concatenate([fc*1.02, (fc*0.98)[::-1]]),
        fill="toself", fillcolor="rgba(245,158,11,0.07)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False, name="±2% band",
    ))
    last_date = df["Date"].max()
    fig.add_shape(type="line", x0=last_date, x1=last_date, y0=0, y1=1,
                  yref="paper", line=dict(color="#45484f", width=1, dash="dot"))
    fig.add_annotation(x=last_date, y=1, yref="paper", text="→ Forecast",
                       showarrow=False, font=dict(color="#a9abb3", size=10), xshift=10)
    fig.update_layout(title=dict(text=f"{ticker} — {model} Forecast",
                                  font=dict(size=13, color="#ecedf6", family="Space Grotesk")), **CT)
    return fig


def _garch_forecast_chart(res, ticker):
    fdf = res["forecast_df"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Forecast"] * 100,
                             name="Volatility Forecast", line=dict(color="#ff6e84", width=2)))
    fig.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Upper"] * 100,
                             name="Upper Band", line=dict(color="rgba(255,110,132,0.4)", width=1), showlegend=False))
    fig.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Lower"] * 100,
                             name="Lower Band", line=dict(color="rgba(255,110,132,0.4)", width=1), fill="tonexty", fillcolor="rgba(255,110,132,0.08)"))
    fig.update_layout(title=dict(text=f"{ticker} — GARCH Volatility Forecast",
                                  font=dict(size=13, color="#ecedf6", family="Space Grotesk")),
                      yaxis_title="Volatility (%)",
                      **CT)
    return fig


def _garch_conditional_chart(res):
    df = res.get("conditional_volatility")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Volatility"] * 100,
                             name="GARCH Conditional Volatility", line=dict(color="#10B981", width=2)))
    fig.update_layout(title=dict(text="GARCH Conditional Volatility", font=dict(size=13, color="#ecedf6", family="Space Grotesk")),
                      yaxis_title="Volatility (%)",
                      **CT)
    return fig


def _model_conditional_vol_chart(feat, model):
    fig = go.Figure()
    if "Vol7" in feat.columns and "Vol30" in feat.columns:
        fig.add_trace(go.Scatter(x=feat["Date"], y=feat["Vol7"] * 100,
                                 name="7d Volatility", line=dict(color="#ff6e84", width=1.8)))
        fig.add_trace(go.Scatter(x=feat["Date"], y=feat["Vol30"] * 100,
                                 name="30d Volatility", line=dict(color="#10B981", width=1.8)))
    else:
        rets = feat["Close"].pct_change()
        fig.add_trace(go.Scatter(x=feat["Date"], y=rets.rolling(7).std() * 100,
                                 name="7d Volatility", line=dict(color="#ff6e84", width=1.8)))
        fig.add_trace(go.Scatter(x=feat["Date"], y=rets.rolling(30).std() * 100,
                                 name="30d Volatility", line=dict(color="#10B981", width=1.8)))

    fig.update_layout(title=dict(text=f"{model} Conditional Volatility", font=dict(size=13, color="#ecedf6", family="Space Grotesk")),
                      yaxis_title="Volatility (%)",
                      **CT)
    return fig


def _xgb_probability_chart(res, ticker):
    fdf = res["forecast_df"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fdf["Date"], y=fdf["Forecast"],
                             name="Up Probability", line=dict(color="#38bdf8", width=2)))
    fig.update_layout(title=dict(text=f"{ticker} — XGBoost Directional Probability", font=dict(size=13, color="#ecedf6", family="Space Grotesk")),
                      yaxis_title="Probability (%)",
                      **CT)
    return fig


def _xgb_probability_gauge(prob_up):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(prob_up) * 100.0,
        title={"text": "Buy Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#38bdf8"},
            "steps": [
                {"range": [0, 35], "color": "rgba(255,110,132,0.25)"},
                {"range": [35, 65], "color": "rgba(245,158,11,0.25)"},
                {"range": [65, 100], "color": "rgba(16,185,129,0.25)"},
            ],
        },
    ))
    fig.update_layout(height=280, **CT)
    return fig


def _xgb_feature_importance_chart(res):
    importance = res.get("feature_importance")
    if importance is None or importance.empty:
        fig = go.Figure()
        fig.update_layout(title=dict(text="XGBoost Feature Importance", font=dict(size=13, color="#ecedf6", family="Space Grotesk")), **CT)
        return fig

    fig = px.bar(importance.head(12), x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Tealgrn")
    fig.update_layout(title=dict(text="Top XGBoost Predictive Features", font=dict(size=13, color="#ecedf6", family="Space Grotesk")),
                      **CT)
    fig.update_yaxes(automargin=True)
    return fig


def _validation_chart(history, title, y, name, color):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["Date"], y=history[y], name=name,
                             line=dict(color=color, width=2)))
    fig.update_layout(title=dict(text=title, font=dict(size=13, color="#ecedf6", family="Space Grotesk")), **CT)
    return fig


def _error_histogram(history):
    fig = px.histogram(history, x="Error %", nbins=30, title="Forecast Error Distribution")
    fig.update_traces(marker_color="#f59e0b")
    fig.update_layout(title=dict(font=dict(size=13, color="#ecedf6", family="Space Grotesk")), **CT)
    return fig


def _accuracy_leaderboard(comparison):
    fig = px.bar(comparison, x="Model", y="Score", text="Score",
                 color="Model", color_discrete_map={"ARIMA":"#f59e0b","Prophet":"#ba9eff","LSTM":"#10B981","GARCH":"#22c55e"})
    fig.update_layout(title=dict(text="Model Comparison Leaderboard", font=dict(size=13, color="#ecedf6", family="Space Grotesk")), showlegend=False, **CT)
    return fig

# ═══════════════════════════════════════════════
# SIGNAL PANEL
# ═══════════════════════════════════════════════════════
def render_signal(sig: dict):
    s     = sig["signal"]
    conf  = sig["confidence"]
    tech  = sig["tech_score"]
    final = sig["final_score"]
    color = "#10B981" if s=="BUY" else "#ff6e84" if s=="SELL" else "#f59e0b"

    st.markdown(f"""
    <div class="sig-wrap sig-{s} sig-glow-{s}">
      <div class="sig-label">{sig['emoji']} {s}</div>
      <div class="sig-sub">AI Confidence: <strong style="color:{color}">{conf}%</strong></div>
    </div>""", unsafe_allow_html=True)

    _bar(conf, color)
    st.markdown(f'<p style="color:#a9abb3;font-size:.7rem;text-align:center">SIGNAL CONFIDENCE</p>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="score-row">
      <div class="score-ring {_sr_cls(tech)}">
        <span class="num">{tech}</span><span class="lbl">Tech</span></div>
      <div class="score-ring {_sr_cls(final)}">
        <span class="num">{final}</span><span class="lbl">Final</span></div>
      <div class="score-ring {_sr_cls(conf)}">
        <span class="num">{conf}</span><span class="lbl">Conf%</span></div>
    </div>""", unsafe_allow_html=True)

    whys  = sig.get("why",  [])
    risks = sig.get("risks", [])

    if whys:
        st.markdown('<p style="color:#a9abb3;font-size:.72rem;margin:10px 0 4px;text-transform:uppercase;letter-spacing:.5px;font-family:Inter,sans-serif">📋 Why this signal</p>',
                    unsafe_allow_html=True)
        for w in whys:
            st.markdown(f'<div class="why-item"><div class="why-dot" style="background:{color}"></div>'
                        f'<div class="why-txt">{w}</div></div>', unsafe_allow_html=True)

    if risks:
        st.markdown('<p style="color:#a9abb3;font-size:.72rem;margin:10px 0 4px;text-transform:uppercase;letter-spacing:.5px;font-family:Inter,sans-serif">⚠️ Risk warnings</p>',
                    unsafe_allow_html=True)
        for r in risks:
            st.markdown(f'<div class="risk-item"><div class="why-dot" style="background:#ff6e84"></div>'
                        f'<div class="risk-txt">{r}</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
def sidebar():
    sb = st.sidebar
    sb.markdown("""
    <div class="sb-logo">
      <div class="sb-logo-icon">🧠</div>
      <div class="sb-logo-name">TradeMind AI</div>
      <div class="sb-logo-ver">v3.0 · AI Trading Assistant</div>
    </div><hr class="sb-divider">
    """, unsafe_allow_html=True)

    sb.markdown('<p style="color:#a9abb3;font-size:.72rem;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;font-family:Inter,sans-serif">Market</p>',
                unsafe_allow_html=True)
    market   = sb.selectbox("Market", list(MARKETS.keys()), label_visibility="collapsed")
    avail    = MARKETS[market]
    selected = sb.multiselect("Assets", avail, default=[avail[0]],
                               label_visibility="collapsed", placeholder="Select assets…")
    custom   = sb.text_input("Custom ticker", placeholder="e.g. NVDA, ETH-USD",
                              label_visibility="collapsed")
    if custom.strip():
        for t in [x.strip().upper() for x in custom.split(",") if x.strip()]:
            if t not in selected:
                selected.append(t)

    sb.markdown("---")
    sb.markdown('<p style="color:#a9abb3;font-size:.72rem;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;font-family:Inter,sans-serif">Time Settings</p>',
                unsafe_allow_html=True)
    c1, c2 = sb.columns(2)
    period   = c1.selectbox("Period",   PERIODS,   index=2, label_visibility="collapsed")
    interval = c2.selectbox("Interval", INTERVALS, index=0, label_visibility="collapsed")
    horizon  = sb.slider("Forecast days", 7, 90, 30, step=7)

    sb.markdown("---")
    if sb.button("🚀  Load & Analyze", use_container_width=True, type="primary"):
        if selected:
            _load(selected, period, interval)
        else:
            sb.error("Select at least one asset.")

    # Watchlist
    sb.markdown("---")
    sb.markdown('<p style="color:#a9abb3;font-size:.72rem;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px;font-family:Inter,sans-serif">⭐ Watchlist</p>',
                unsafe_allow_html=True)
    wl = sb.text_input("Add ticker", placeholder="BTC-USD", label_visibility="collapsed")
    if sb.button("Add to watchlist", use_container_width=True):
        t = wl.strip().upper()
        if t and t not in st.session_state.watchlist:
            st.session_state.watchlist.append(t)
    for w in st.session_state.watchlist:
        ca, cb = sb.columns([4, 1])
        ca.markdown(f'<span style="color:#a9abb3;font-size:.83rem;font-family:JetBrains Mono,monospace">{w}</span>',
                    unsafe_allow_html=True)
        if cb.button("✕", key=f"wl_{w}"):
            st.session_state.watchlist.remove(w); st.rerun()

    sb.markdown("---")
    sb.caption("⚠️ Educational only · Not financial advice")
    return selected, period, interval, horizon


def _load(tickers, period, interval):
    st.session_state.data = {} # Clear stale data
    st.session_state.last_forecast = {}
    st.session_state.last_garch = {}
    st.session_state.last_ml = {}
    st.session_state.last_rl = {}
    bar    = st.sidebar.progress(0)
    status = st.sidebar.empty()
    loaded = 0
    for i, t in enumerate(tickers):
        status.markdown(f'<span style="color:#a9abb3;font-size:.8rem">⏳ {t}…</span>',
                        unsafe_allow_html=True)
        bar.progress((i+1)/len(tickers))
        df = load_asset(t, period, interval)
        if df.empty:
            st.sidebar.error(f"❌ Asset not found: {t}")
            continue
        try:
            feat = add_all(df)
            sig  = generate(feat, rl_info=st.session_state.last_rl.get(t, {}))
            st.session_state.data[t] = {"df": df, "feat": feat, "sig": sig}
            loaded += 1
        except Exception as e:
            st.sidebar.error(f"{t}: {e}")
    bar.empty()
    if loaded:
        status.markdown(f'<span style="color:#10B981;font-size:.85rem">✅ {loaded} asset(s) ready!</span>',
                        unsafe_allow_html=True)
    else:
        status.error("No data loaded.")

# ═══════════════════════════════════════════════════════
# SIGNAL BANNER — all loaded assets
# ═══════════════════════════════════════════════════════
def signal_banner():
    data = st.session_state.data
    if len(data) < 2:
        return
    st.markdown("### 🔥 Signal Overview")
    cells = ""
    for t, obj in data.items():
        s   = obj["sig"]
        clr = "#10B981" if s["signal"]=="BUY" else "#ff6e84" if s["signal"]=="SELL" else "#f59e0b"
        cells += (f'<div class="opp-card"><div class="opp-ticker">{t}</div>'
                  f'<div class="opp-sig" style="color:{clr}">{s["emoji"]} {s["signal"]}</div>'
                  f'<div class="opp-conf">{s["confidence"]}% confidence</div></div>')
    st.markdown(f'<div class="opp-grid">{cells}</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════

# ── Overview ──────────────────────────────────────────
def tab_overview(df, feat, ticker, curr):
    last  = float(df["Close"].iloc[-1])
    prev  = float(df["Close"].iloc[-2]) if len(df)>1 else last
    chg   = (last-prev)/prev*100
    n     = min(252, len(df))
    h52   = float(df["Close"].rolling(n).max().iloc[-1])
    l52   = float(df["Close"].rolling(n).min().iloc[-1])
    vol   = float(df["Close"].pct_change().std()*np.sqrt(252)*100)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Last Price",     f"{curr}{last:,.4f}", f"{chg:+.2f}%")
    c2.metric("52W High",       f"{curr}{h52:,.2f}")
    c3.metric("52W Low",        f"{curr}{l52:,.2f}")
    c4.metric("Ann. Volatility",f"{vol:.1f}%")
    c5.metric("Data Points",    f"{len(df):,}")

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)

    col_chart, col_sig = st.columns([3, 1], gap="large")
    with col_chart:
        fig = _candle(df, f"{ticker} — Price History", tail=120)
        # Add volume bars at bottom if available
        if "Volume" in df.columns:
            df_t = df.tail(120)
            colors = ["rgba(16, 185, 129, 0.5)" if df_t["Close"].iloc[i] >= df_t["Open"].iloc[i]
                      else "rgba(255, 110, 132, 0.5)" for i in range(len(df_t))]
            fig.add_trace(go.Bar(x=df_t["Date"], y=df_t["Volume"],
                                  name="Volume", marker_color=colors,
                                  opacity=0.3, yaxis="y2"))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right",
                                          showgrid=False, showticklabels=False,
                                          range=[0, df_t["Volume"].max()*8]))
        st.plotly_chart(fig, use_container_width=True)
    with col_sig:
        sig = st.session_state.data[ticker]["sig"]
        render_signal(sig)

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
    st.markdown("**📊 Indicator Snapshot**")
    tbl = indicator_table(feat)
    st.dataframe(tbl, use_container_width=True, hide_index=True)


# ── Raw Data ───────────────────────────────────────────
def tab_raw(df, ticker, curr):
    c1,c2,c3 = st.columns(3)
    c1.metric("Rows",  f"{len(df):,}")
    c2.metric("From",  str(df["Date"].min().date()))
    c3.metric("To",    str(df["Date"].max().date()))
    st.dataframe(df.tail(50), use_container_width=True, hide_index=True)
    t1,t2 = st.tabs(["Line","Candlestick"])
    with t1:
        fig = go.Figure(go.Scatter(x=df["Date"],y=df["Close"],
                                   line=dict(color="#0EA5E9",width=2),name="Close",
                                   fill="tozeroy",fillcolor="rgba(14, 165, 233, 0.04)"))
        fig.update_layout(title=dict(text=f"{ticker} Close",font=dict(size=13,color="#ecedf6", family="Space Grotesk")),**CT)
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        st.plotly_chart(_candle(df,f"{ticker} OHLC"), use_container_width=True)


# ── Indicators ─────────────────────────────────────────
def tab_indicators(feat):
    # RSI + MACD
    c1,c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=feat["Date"],y=feat["RSI"],
                                 name="RSI",line=dict(color="#0EA5E9",width=2)))
        fig.add_hrect(y0=70,y1=100,fillcolor="rgba(255, 110, 132, 0.06)",line_width=0)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(16, 185, 129, 0.06)",line_width=0)
        fig.add_hline(y=70,line_color="#ff6e84",line_dash="dot",line_width=1,
                      annotation_text="Overbought",annotation_font_color="#ff6e84",annotation_font_size=10)
        fig.add_hline(y=30,line_color="#10B981",line_dash="dot",line_width=1,
                      annotation_text="Oversold",annotation_font_color="#10B981",annotation_font_size=10)
        fig.update_layout(title=dict(text="RSI (14)",font=dict(size=13,color="#ecedf6", family="Space Grotesk")),
                          yaxis_range=[0,100],**CT)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        clrs = feat["MACD_hist"].apply(lambda v:"#10B981" if v>=0 else "#ff6e84")
        fig  = go.Figure()
        fig.add_bar(x=feat["Date"],y=feat["MACD_hist"],name="Histogram",marker_color=clrs,opacity=.7)
        fig.add_trace(go.Scatter(x=feat["Date"],y=feat["MACD"],
                                 name="MACD",line=dict(color="#0EA5E9",width=1.5)))
        fig.add_trace(go.Scatter(x=feat["Date"],y=feat["MACD_sig"],
                                 name="Signal",line=dict(color="#f59e0b",width=1.5)))
        fig.update_layout(title=dict(text="MACD (12,26,9)",font=dict(size=13,color="#ecedf6", family="Space Grotesk")),**CT)
        st.plotly_chart(fig, use_container_width=True)

    # Bollinger + Stochastic
    c3,c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=feat["Date"],y=feat["BB_upper"],
                                 name="Upper",line=dict(color="rgba(69, 72, 79, 0.6)",width=1)))
        fig.add_trace(go.Scatter(x=feat["Date"],y=feat["BB_lower"],
                                 name="Lower",line=dict(color="rgba(69, 72, 79, 0.6)",width=1),
                                 fill="tonexty",fillcolor="rgba(14, 165, 233, 0.05)"))
        fig.add_trace(go.Scatter(x=feat["Date"],y=feat["Close"],
                                 name="Price",line=dict(color="#0EA5E9",width=2)))
        fig.add_trace(go.Scatter(x=feat["Date"],y=feat["BB_mid"],
                                 name="Mid",line=dict(color="#ecedf6",width=1,dash="dot")))
        fig.update_layout(title=dict(text="Bollinger Bands (20,2)",font=dict(size=13,color="#ecedf6", family="Space Grotesk")),**CT)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=feat["Date"],y=feat["Stoch_K"],
                                 name="%K",line=dict(color="#ba9eff",width=1.5)))
        fig.add_trace(go.Scatter(x=feat["Date"],y=feat["Stoch_D"],
                                 name="%D",line=dict(color="#f59e0b",width=1.5)))
        fig.add_hline(y=80,line_color="#ff6e84",line_dash="dot",line_width=1)
        fig.add_hline(y=20,line_color="#10B981",line_dash="dot",line_width=1)
        fig.update_layout(title=dict(text="Stochastic (14,3)",font=dict(size=13,color="#ecedf6", family="Space Grotesk")),
                          yaxis_range=[0,100],**CT)
        st.plotly_chart(fig, use_container_width=True)

    # Volatility
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=feat["Date"],y=feat["Vol7"]*100,
                             name="7d Vol%",line=dict(color="#ff6e84",width=1.5),
                             fill="tozeroy",fillcolor="rgba(255, 110, 132, 0.05)"))
    fig.add_trace(go.Scatter(x=feat["Date"],y=feat["Vol30"]*100,
                             name="30d Vol%",line=dict(color="#f59e0b",width=1.5)))
    fig.update_layout(title=dict(text="Rolling Volatility (%)",font=dict(size=13,color="#ecedf6", family="Space Grotesk")),**CT)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
    st.markdown("**Indicator Summary**")
    st.dataframe(indicator_table(feat), use_container_width=True, hide_index=True)


# ── Patterns ───────────────────────────────────────────
def tab_patterns(feat):
    pats = feat[feat.get("Pattern", pd.Series("",index=feat.index)) != ""]
    if pats.empty:
        st.info("No patterns detected. Try a longer period (2y+)."); return
    st.success(f"✅ {len(pats)} pattern(s) detected")
    c1,c2 = st.columns([1,2])
    with c1:
        cols = [c for c in ["Date","Close","Pattern"] if c in pats.columns]
        st.dataframe(pats[cols].tail(20), use_container_width=True, hide_index=True)
    with c2:
        pc = pats["Pattern"].value_counts().reset_index()
        pc.columns = ["Pattern","Count"]
        fig = px.bar(pc,x="Pattern",y="Count",color="Count",
                     color_continuous_scale=[[0,"#10131a"],[1,"#0EA5E9"]],
                     title="Pattern Frequency")
        fig.update_layout(title=dict(font=dict(size=13,color="#ecedf6", family="Space Grotesk")),**CT); st.plotly_chart(fig, use_container_width=True)
    if all(c in feat.columns for c in ["Open","High","Low"]):
        fig = _candle(feat.tail(120),"Patterns on Chart")
        if "High" in pats.columns:
            fig.add_trace(go.Scatter(
                x=pats.tail(30)["Date"],y=pats.tail(30)["High"]*1.015,
                mode="markers+text",text=pats.tail(30)["Pattern"],
                textposition="top center",
                marker=dict(color="#f59e0b",size=10,symbol="triangle-up"),name="Pattern",
            ))
        st.plotly_chart(fig, use_container_width=True)


# ── Forecasting ────────────────────────────────────────
def tab_forecast(df, feat, ticker, horizon):
    st.markdown("""
    <div style="background:rgba(34, 38, 47, 0.4);border:1px solid rgba(255,255,255,0.1);border-radius:10px;
         padding:12px 16px;margin-bottom:16px;backdrop-filter: blur(20px);">
      <p style="color:#10B981;font-size:.8rem;margin:0;font-family:Inter,sans-serif">
        ✅ <strong>Bug fixed:</strong> ARIMA / Prophet / LSTM now use RangeIndex internally.
        Zero date arithmetic inside models — the int+datetime error is permanently eliminated.
      </p>
    </div>""", unsafe_allow_html=True)

    mode = st.radio("Mode", ["Single Model","Compare All Models"], horizontal=True,
                    label_visibility="collapsed")

    if mode == "Single Model":
        if len(df) < 30:
            st.error("⚠️ Insufficient data for forecasting (needs at least 30 periods).")
            return
        c1,c2 = st.columns([2,1])
        model  = c1.selectbox("Model", ["ARIMA","Prophet","LSTM","GARCH","XGBoost"])
        run    = c2.button("▶ Run Forecast", type="primary", use_container_width=True)

        if run:
            with st.spinner(f"Running {model}… (LSTM may take ~60s)" if model == "LSTM" else f"Running {model}…"):
                try:
                    if model == "XGBoost":
                        res = _xgboost.forecast(df, horizon, sentiment_features=st.session_state.get("last_sentiment", {}))
                    else:
                        res = {"ARIMA":_arima,"Prophet":_prophet,"LSTM":_lstm,"GARCH":_garch}[model].forecast(df,horizon)
                    fdf = res["forecast_df"]
                    st.session_state.last_forecast[ticker] = fdf
                    if model == "GARCH":
                        st.session_state.last_garch[ticker] = res
                        st.session_state.last_ml[ticker] = {}
                        sig = generate(feat, fdf, garch_info=res, rl_info=st.session_state.last_rl.get(ticker, {}))
                    elif model == "XGBoost":
                        st.session_state.last_garch[ticker] = {}
                        st.session_state.last_ml[ticker] = res
                        sig = generate(feat, fdf, ml_info=res, rl_info=st.session_state.last_rl.get(ticker, {}))
                    else:
                        st.session_state.last_garch[ticker] = {}
                        st.session_state.last_ml[ticker] = {}
                        sig = generate(feat, fdf, rl_info=st.session_state.last_rl.get(ticker, {}))
                    st.session_state.data[ticker]["sig"] = sig

                    m = res["metrics"]
                    if model == "XGBoost":
                        c1,c2,c3 = st.columns(3)
                        c1.metric("Accuracy", f"{m.get('Accuracy', np.nan):.4f}")
                        c2.metric("F1 Score", f"{m.get('F1', np.nan):.4f}")
                        c3.metric("ROC-AUC", f"{m.get('ROC-AUC', np.nan):.4f}")
                        c4,c5 = st.columns(2)
                        c4.metric("Precision", f"{m.get('Precision', np.nan):.4f}")
                        c5.metric("Recall", f"{m.get('Recall', np.nan):.4f}")
                    else:
                        c1,c2,c3 = st.columns(3)
                        c1.metric("RMSE (normalized)", f"{m.get('RMSE (norm)', np.nan):.4f}")
                        c2.metric("MAE (normalized)",  f"{m.get('MAE (norm)', np.nan):.4f}")
                        c3.metric("MAPE",              f"{m.get('MAPE (%)', np.nan):.2f}%")

                    vols = {
                        "Realized 7d": feat["Vol7"].iloc[-1] if "Vol7" in feat.columns else np.nan,
                        "Realized 30d": feat["Vol30"].iloc[-1] if "Vol30" in feat.columns else np.nan,
                    }
                    regime_label = "High volatility" if vols["Realized 30d"] >= 0.03 else "Moderate volatility" if vols["Realized 30d"] >= 0.012 else "Low volatility"
                    with st.expander("📌 Volatility Insights", expanded=True):
                        vc1,vc2,vc3 = st.columns(3)
                        vc1.metric("Realized 7d Vol", f"{vols['Realized 7d']*100:.2f}%")
                        vc2.metric("Realized 30d Vol", f"{vols['Realized 30d']*100:.2f}%")
                        vc3.metric("Volatility Regime", regime_label)
                        if model == "GARCH":
                            vol = res["vol_forecast"]
                            st.markdown(f"**Forecasted Volatility:** 7d {vol['7d']*100:.2f}% · 14d {vol['14d']*100:.2f}% · 30d {vol['30d']*100:.2f}%")
                            for warning in res.get("warnings", []):
                                st.warning(warning)
                        else:
                            st.markdown("Model forecasts are shown alongside realized volatility. Use high volatility regimes to temper signal confidence.")

                    if model == "GARCH":
                        st.plotly_chart(_garch_forecast_chart(res, ticker), use_container_width=True)
                        st.plotly_chart(_garch_conditional_chart(res), use_container_width=True)
                    elif model == "XGBoost":
                        st.info("📌 XGBoost predicts directional probability rather than exact price. Use this as a strong ML confirmation signal.")
                        prob = res.get("probability", 0.5)
                        pred = res.get("prediction", "Down")
                        lbl = res.get("label", "Bearish")
                        tone = "#10B981" if pred == "Up" else "#ff6e84"
                        st.markdown(
                            f'<div style="background:rgba(34, 38, 47, 0.4);border:1px solid rgba(255,255,255,0.1);border-radius:12px;padding:14px 16px;margin-bottom:12px;">'
                            f'<div style="color:#a9abb3;font-size:.8rem;">Direction Prediction</div>'
                            f'<div style="color:{tone};font-family:Space Grotesk;font-weight:700;font-size:1.25rem;">{pred} ({lbl})</div>'
                            f'<div style="color:#ecedf6;font-size:.85rem;">Buy {prob*100:.1f}% · Sell {(1-prob)*100:.1f}%</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(_xgb_probability_gauge(prob), use_container_width=True)
                        st.plotly_chart(_xgb_probability_chart(res, ticker), use_container_width=True)
                        st.plotly_chart(_xgb_feature_importance_chart(res), use_container_width=True)
                    else:
                        st.info("📌 Normalized RMSE/MAE are scaled to [0,1]. Values <0.05 indicate good fit. MAPE shows % prediction error.")
                        st.plotly_chart(_forecast_chart(df, fdf, model, ticker), use_container_width=True)
                        st.plotly_chart(_model_conditional_vol_chart(feat, model), use_container_width=True)

                    if model == "XGBoost":
                        prob = res.get("probability", 0.0)
                        st.markdown(f"**Directional prediction:** {res.get('prediction')} / {res.get('label')}\n\n")
                        st.markdown(f"**Buy probability:** {prob*100:.1f}% · **Sell probability:** {(1-prob)*100:.1f}%")
                        st.markdown("### Confusion Matrix")
                        st.dataframe(res.get("confusion_matrix", pd.DataFrame()), use_container_width=True, hide_index=True)
                    else:
                        with st.expander("📋 Forecast Table"):
                            st.dataframe(fdf, use_container_width=True, hide_index=True)

                except Exception as e:
                    st.error(f"Forecast failed: {e}")
                    with st.expander("Debug info"):
                        import traceback; st.code(traceback.format_exc())
    else:
        if st.button("▶ Compare All Models", type="primary"):
            with st.spinner("Running ARIMA + Prophet + LSTM…"):
                out = run_all_models(df, horizon)
            if out["errors"]:
                for n,e in out["errors"].items(): st.warning(f"⚠️ {n}: {e}")
            if out["comparison"].empty:
                st.error("No models ran successfully."); return
            st.success(f"🏆 Best model: **{out['best_model']}**")
            st.dataframe(out["comparison"], use_container_width=True, hide_index=True)

            fdf = out["results"][out["best_model"]]["forecast_df"]
            st.session_state.last_forecast[ticker] = fdf
            garch_info = out["results"][out["best_model"]] if out["best_model"] == "GARCH" else None
            ml_info = out["results"][out["best_model"]] if out["best_model"] == "XGBoost" else None
            st.session_state.last_garch[ticker] = out["results"][out["best_model"]] if out["best_model"] == "GARCH" else {}
            st.session_state.last_ml[ticker] = out["results"][out["best_model"]] if out["best_model"] == "XGBoost" else {}
            sig = generate(feat, fdf, garch_info=garch_info, ml_info=ml_info, rl_info=st.session_state.last_rl.get(ticker, {}))
            st.session_state.data[ticker]["sig"] = sig

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"],y=df["Close"],
                                     name="Historical",line=dict(color="#0EA5E9",width=2)))
            pal = {"ARIMA":"#f59e0b","Prophet":"#ba9eff","LSTM":"#ff6e84","GARCH":"#22c55e","XGBoost":"#38bdf8"}
            has_garch = any(n == "GARCH" for n in out["results"].keys())
            for n,r in out["results"].items():
                if n == "GARCH":
                    fig.add_trace(go.Scatter(x=r["forecast_df"]["Date"], y=r["forecast_df"]["Forecast"] * 100,
                                             name="GARCH Vol Forecast", line=dict(color=pal[n], dash="dot", width=1.5),
                                             yaxis="y2"))
                elif n == "XGBoost":
                    fig.add_trace(go.Scatter(x=r["forecast_df"]["Date"], y=r["forecast_df"]["Forecast"],
                                             name="XGBoost Up Probability", line=dict(color=pal[n], dash="dot", width=1.5),
                                             yaxis="y2"))
                else:
                    fig.add_trace(go.Scatter(x=r["forecast_df"]["Date"],y=r["forecast_df"]["Forecast"],
                                             name=n,line=dict(color=pal.get(n,"#fff"),dash="dash",width=1.5)))
            if has_garch or any(n == "XGBoost" for n in out["results"].keys()):
                fig.update_layout(yaxis2=dict(
                    overlaying="y",
                    side="right",
                    title=dict(text="Volatility / Probability (%)", font=dict(color="#22c55e")),
                    showgrid=False,
                    tickfont=dict(color="#22c55e"),
                ))
            fig.update_layout(title=dict(text=f"{ticker} — Model Comparison",
                                          font=dict(size=13,color="#ecedf6", family="Space Grotesk")),**CT)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
    st.markdown("### ⚡ AI Trading Signal")
    fdf_now = st.session_state.last_forecast.get(ticker)
    garch_info = st.session_state.last_garch.get(ticker)
    ml_info = st.session_state.last_ml.get(ticker)
    rl_info = st.session_state.last_rl.get(ticker)
    sig = generate(feat, fdf_now, garch_info=garch_info, ml_info=ml_info, rl_info=rl_info)
    st.session_state.data[ticker]["sig"] = sig

    col_s, col_c = st.columns([1,2])
    with col_s:
        render_signal(sig)
    with col_c:
        st.markdown("**Component Breakdown**")
        rows = [{"Component":n,
                 "Weight":v["score"],
                 "Direction":"🟢 Bull" if v["dir"]>0 else "🔴 Bear" if v["dir"]<0 else "🟡 Neutral"}
                for n,v in sig["components"].items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if st.button("🔔 Send Alert", use_container_width=True):
            st.session_state.alerts.signal_alert(
                ticker, sig["signal"], sig["confidence"], sig.get("why",[]))
            st.success("Alert sent!")


def tab_forecast_accuracy(df, ticker):
    st.markdown("""
    <div style="background:rgba(34, 38, 47, 0.4);border:1px solid rgba(255,255,255,0.1);border-radius:10px;
         padding:16px 18px;margin-bottom:16px;backdrop-filter: blur(20px);">
      <p style="color:#ba9eff;font-size:.85rem;margin:0;font-family:Inter,sans-serif">
                🧪 Walk-forward forecast validation tests historical accuracy for ARIMA, Prophet, LSTM, and XGBoost.
        Compare predicted vs actual price behavior across rolling and expanding windows.
      </p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3 = st.columns(3)
    model = c1.selectbox(
    "Model",
    ["ARIMA", "Prophet", "LSTM", "XGBoost"],
    key="forecast_model_select"
)
    horizon = c2.selectbox("Horizon", [7, 14, 30], index=0)
    method = c3.selectbox("Walk-forward Method", ["Expanding","Rolling"], index=0)

    step = st.slider("Step size", 1, 30, 14, step=1)
    rolling_window = DEFAULT_ROLLING_WINDOW if method == "Rolling" else None
    if method == "Rolling":
        rolling_window = st.slider("Rolling window", 60, 360, 180, step=10)

    threshold = st.slider("Signal threshold (%)", 1, 10, 2, step=1)
    run = st.button("▶ Run Validation", type="primary", use_container_width=True)

    if not run:
        st.info("Run walk-forward validation to compare forecast accuracy and simulated forecast-driven strategy performance.")
        return

    with st.spinner("Running walk-forward validation…"):
        try:
            result = walk_forward_validate(
                df, model, horizon,
                method=method.lower(),
                step=step,
                rolling_window=rolling_window or DEFAULT_ROLLING_WINDOW,
                min_train_size=max(60, horizon * 2),
                hit_threshold=5.0,
                direction_threshold_pct=float(threshold),
            )

            compare = compare_models_accuracy(
                df, horizon,
                method=method.lower(),
                step=step,
                rolling_window=rolling_window or DEFAULT_ROLLING_WINDOW,
                min_train_size=max(60, horizon * 2),
                hit_threshold=5.0,
                direction_threshold_pct=float(threshold),
                precomputed_summaries={model: result["summary"]},
            )

            strategy = simulate_forecast_strategy(result["history"], threshold_pct=threshold, capital=10000.0)
        except Exception as e:
            st.error(f"Validation failed: {e}")
            with st.expander("Debug info"):
                import traceback; st.code(traceback.format_exc())
            return

    summary = result["summary"]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Direction Accuracy", f"{summary['Direction Accuracy']:.2f}%")
    c2.metric("Forecast Hit Rate", f"{summary['Forecast Hit Rate']:.2f}%")
    c3.metric("Avg Error", f"{summary['Avg Error %']:.2f}%")
    c4.metric("MAPE", f"{summary['MAPE']:.2f}%")
    c5.metric("Confidence", f"{summary['Forecast Confidence']}%")

    st.markdown(f"**Best model overall:** {compare['best_model'] or 'N/A'}")
    if compare["errors"]:
        for k,v in compare["errors"].items():
            st.warning(f"⚠️ {k}: {v}")

    st.markdown('---')
    st.markdown("### Model Comparison Leaderboard")
    if not compare["comparison"].empty:
        st.dataframe(compare["comparison"], use_container_width=True, hide_index=True)
        st.plotly_chart(_accuracy_leaderboard(compare["comparison"]), use_container_width=True)

    st.markdown('---')
    st.markdown("### Forecast vs Actual")
    st.plotly_chart(_validation_chart(result["history"], "Predicted vs Actual End Price", "Actual", "Actual", "#0EA5E9"), use_container_width=True)
    st.plotly_chart(_validation_chart(result["history"], "Predicted vs Actual End Price", "Predicted", "Predicted", "#f59e0b"), use_container_width=True)

    st.markdown('---')
    c1,c2 = st.columns(2)
    with c1:
        st.plotly_chart(_validation_chart(result["history"], "Rolling Forecast Error", "Error %", "Error %", "#ff6e84"), use_container_width=True)
    with c2:
        if not compare["comparison"].empty:
            st.plotly_chart(
    _accuracy_leaderboard(compare["comparison"]),
    use_container_width=True,
    key="accuracy_leaderboard_chart"
)

    st.plotly_chart(_error_histogram(result["history"]), use_container_width=True)

    if strategy:
        st.markdown('---')
        st.markdown("### Forecast-driven Strategy Simulation")
        m = strategy["metrics"]
        sc1,sc2,sc3,sc4,sc5 = st.columns(5)
        sc1.metric("Return", f"{m['Total Return %']:+.2f}%")
        sc2.metric("Win Rate", f"{m['Win Rate %']:.2f}%")
        sc3.metric("Sharpe", f"{m['Sharpe']:.2f}")
        sc4.metric("Max Drawdown", f"{m['Max Drawdown %']:.2f}%")
        sc5.metric("Profit Factor", f"{m['Profit Factor']:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strategy["equity_curve"]["Date"], y=strategy["equity_curve"]["Equity"],
                                 name="Forecast Strategy", line=dict(color="#10B981", width=2)))
        fig.update_layout(title=dict(text="Forecast Strategy Equity Curve", font=dict(size=13, color="#ecedf6", family="Space Grotesk")), **CT)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Forecast Strategy Trades"):
            st.dataframe(strategy["trades"].tail(50), use_container_width=True, hide_index=True)

    st.markdown('---')
    st.markdown("**Walk-forward validation history sample**")
    st.dataframe(result["history"].tail(20), use_container_width=True, hide_index=True)


# ── Backtest ───────────────────────────────────────────
def tab_backtest(feat, ticker, curr):
    c1,c2,c3 = st.columns(3)
    strategy = c1.selectbox("Strategy", STRATEGY_NAMES)
    capital  = c2.number_input("Capital", value=10000, step=1000, min_value=100)
    run = c3.button("▶ Run Backtest", type="primary", use_container_width=True)

    if not run:
        st.info("Select a strategy and click **Run Backtest** to simulate on historical data."); return

    with st.spinner("Running backtest…"):
        # Prepare backtest data: merge forecast if available for forecast-based strategies
        backtest_feat = feat.copy()
        if "Forecast" in strategy and ticker in st.session_state.get("last_forecast", {}):
            fdf = st.session_state.last_forecast[ticker]
            # Merge forecast prices into the feature dataframe
            if "Forecast" in fdf.columns:
                backtest_feat = backtest_feat.merge(
                    fdf[["Date", "Forecast"]], on="Date", how="left"
                )
                # Forward-fill any missing forecast values
                backtest_feat["Forecast"] = backtest_feat["Forecast"].ffill()
        
        result = run_backtest(backtest_feat, strategy, float(capital))
    m  = result["metrics"]; bh = result["buy_hold_return"]
    eq = result["equity_curve"]; log = result["trade_log"]

    st.markdown("---")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Return",  f"{m['Total Return %']:+.1f}%", f"B&H: {bh:+.1f}%")
    c2.metric("Win Rate",      f"{m['Win Rate %']:.0f}%",
              f"{m['Win Trades']}W/{m['Loss Trades']}L")
    c3.metric("Max Drawdown",  f"{m['Max Drawdown %']:.1f}%")
    c4.metric("Sharpe",        f"{m['Sharpe Ratio']:.2f}")
    c5.metric("Profit Factor", f"{m['Profit Factor']:.2f}")

    c1b,c2b,c3b,c4b = st.columns(4)
    c1b.metric("Avg Win",     f"{m['Avg Win %']:+.2f}%")
    c2b.metric("Avg Loss",    f"{m['Avg Loss %']:.2f}%")
    c3b.metric("Total Trades",str(m['Total Trades']))
    c4b.metric("Final Equity",f"{curr}{m['Final Equity']:,.0f}")

    # Equity curve
    fig = go.Figure()
    if not eq.empty:
        fig.add_trace(go.Scatter(x=eq["Date"],y=eq["Equity"],
                                 name="Strategy",fill="tozeroy",
                                 line=dict(color="#10B981",width=2),
                                 fillcolor="rgba(16, 185, 129, 0.05)"))
    bh_eq = capital * feat["Close"] / feat["Close"].iloc[0]
    fig.add_trace(go.Scatter(x=feat["Date"],y=bh_eq,name="Buy & Hold",
                             line=dict(color="#45484f",width=1.5,dash="dot")))
    fig.update_layout(title=dict(text=f"{strategy} — Equity vs Buy & Hold",
                                  font=dict(size=13,color="#ecedf6", family="Space Grotesk")),
                      yaxis_title="Portfolio Value", **CT)
    st.plotly_chart(fig, use_container_width=True)

    if not log.empty:
        with st.expander(f"📋 Trade Log ({len(log)} trades)"):
            st.dataframe(log, use_container_width=True, hide_index=True)


def tab_rl_agent_lab(df, feat, ticker, curr, horizon):
    st.markdown("### 🤖 RL Agent Lab")
    st.caption("Experimental Q-learning trading agent using indicators, XGBoost probability, and GARCH regime context.")

    c1, c2, c3, c4 = st.columns(4)
    episodes = c1.slider("Episodes", 20, 200, 80, step=10)
    alpha = c2.slider("Learning Rate (alpha)", 0.05, 0.5, 0.15, step=0.05)
    gamma = c3.slider("Discount (gamma)", 0.70, 0.99, 0.95, step=0.01)
    epsilon = c4.slider("Exploration (epsilon)", 0.05, 0.60, 0.25, step=0.05)

    run = st.button("▶ Train RL Agent", type="primary", use_container_width=True)
    cache_key = f"{ticker}|{episodes}|{alpha}|{gamma}|{epsilon}|{horizon}"

    if run:
        with st.spinner("Training Q-learning agent and running strategy benchmarks…"):
            try:
                out = _rl.train_q_agent(
                    df,
                    episodes=episodes,
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=epsilon,
                    sentiment_features=st.session_state.get("last_sentiment", {}),
                )
                st.session_state.rl_lab_cache[cache_key] = out
                st.session_state.last_rl[ticker] = out.get("signal_source", {"action": "HOLD", "confidence": 50.0})
            except Exception as e:
                st.error(f"RL training failed: {e}")
                with st.expander("Debug info"):
                    import traceback; st.code(traceback.format_exc())
                return

    if cache_key not in st.session_state.rl_lab_cache:
        st.info("Train the RL agent to view actions, learning curves, benchmark comparison, and RL signal output.")
        return

    out = st.session_state.rl_lab_cache[cache_key]
    m = out.get("metrics", {})
    s = out.get("signal_source", {"action": "HOLD", "confidence": 50.0})

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("RL Return", f"{m.get('Total Return %', np.nan):+.2f}%")
    r2.metric("RL Sharpe", f"{m.get('Sharpe', np.nan):.3f}" if pd.notna(m.get("Sharpe", np.nan)) else "N/A")
    r3.metric("RL Max DD", f"{m.get('Max Drawdown %', np.nan):.2f}%" if pd.notna(m.get("Max Drawdown %", np.nan)) else "N/A")
    r4.metric("RL Win Rate", f"{m.get('Win Rate %', np.nan):.2f}%")
    r5.metric("RL Signal", f"{s.get('action', 'HOLD')} ({s.get('confidence', 50.0):.0f}%)")

    comp = out.get("comparison", pd.DataFrame())
    if not comp.empty:
        st.markdown("#### RL Strategy Backtesting Comparison")
        st.dataframe(comp, use_container_width=True, hide_index=True)

    eq = out.get("equity_comparison", pd.DataFrame())
    if isinstance(eq, pd.DataFrame) and not eq.empty and "Date" in eq.columns:
        fig = go.Figure()
        for col in eq.columns:
            if col == "Date":
                continue
            fig.add_trace(go.Scatter(x=eq["Date"], y=eq[col], name=col, line=dict(width=2 if col == "RL Equity" else 1.6)))
        fig.update_layout(
            title=dict(text=f"{ticker} — RL vs Benchmarks Equity Curve", font=dict(size=13, color="#ecedf6", family="Space Grotesk")),
            yaxis_title="Normalized Equity",
            **CT,
        )
        st.plotly_chart(fig, use_container_width=True)

    actions = out.get("actions", pd.DataFrame())
    learning = out.get("learning_curve", pd.DataFrame())
    trades = out.get("trades", pd.DataFrame())

    if isinstance(actions, pd.DataFrame) and not actions.empty:
        st.markdown("#### Agent Actions Over Time")
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=actions["Date"], y=actions["Close"], name="Close", line=dict(color="#0EA5E9", width=1.8)))
        buys = actions[actions["Action"] == "BUY"]
        sells = actions[actions["Action"] == "SELL"]
        if not buys.empty:
            f2.add_trace(go.Scatter(x=buys["Date"], y=buys["Close"], mode="markers", marker=dict(color="#10B981", size=8, symbol="triangle-up"), name="BUY"))
        if not sells.empty:
            f2.add_trace(go.Scatter(x=sells["Date"], y=sells["Close"], mode="markers", marker=dict(color="#ff6e84", size=8, symbol="triangle-down"), name="SELL"))
        f2.update_layout(title=dict(text="Agent Trade Decisions", font=dict(size=13, color="#ecedf6", family="Space Grotesk")), **CT)
        st.plotly_chart(f2, use_container_width=True)

    if isinstance(learning, pd.DataFrame) and not learning.empty:
        st.markdown("#### Learning / Reward Curve")
        f3 = go.Figure()
        f3.add_trace(go.Scatter(x=learning["Episode"], y=learning["Total Reward"], name="Total Reward", line=dict(color="#ba9eff", width=2)))
        f3.update_layout(title=dict(text="Q-Learning Episode Rewards", font=dict(size=13, color="#ecedf6", family="Space Grotesk")),
                         xaxis_title="Episode", yaxis_title="Reward", **CT)
        st.plotly_chart(f3, use_container_width=True)

    cta, ctb = st.columns(2)
    with cta:
        if isinstance(trades, pd.DataFrame) and not trades.empty:
            with st.expander("RL Trade Log"):
                st.dataframe(trades, use_container_width=True, hide_index=True)


# ── Risk ───────────────────────────────────────────────
def tab_risk(df, ticker):
    metrics = risk_compute(df, st.session_state.last_garch.get(ticker))
    if not metrics:
        st.warning("Not enough data for risk calculation."); return

    items = list(metrics.items())
    for i in range(0, len(items), 4):
        cols = st.columns(4)
        for j,(k,v) in enumerate(items[i:i+4]):
            if v is not None: cols[j].metric(k, f"{v:.4f}" if isinstance(v,float) else str(v))

    sr = metrics.get("Sharpe Ratio")
    if sr: st.info(f"**Sharpe Assessment:** {sharpe_label(sr)}")

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    close = df["Close"].dropna(); rets = close.pct_change().dropna()
    with c1:
        cum = (1+rets).cumprod(); peak = cum.cummax(); dd = (cum-peak)/peak*100
        fig = go.Figure(go.Scatter(x=df["Date"].iloc[1:],y=dd,fill="tozeroy",
                                   line=dict(color="#ff6e84"),fillcolor="rgba(255, 110, 132, 0.08)",name="DD%"))
        fig.update_layout(title=dict(text="Drawdown (%)",font=dict(size=13,color="#ecedf6", family="Space Grotesk")),**CT)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(rets*100,nbins=60,title="Return Distribution (%)",
                           color_discrete_sequence=["#0EA5E9"])
        fig.update_layout(title=dict(font=dict(size=13,color="#ecedf6", family="Space Grotesk")),**CT); st.plotly_chart(fig, use_container_width=True)

    # Rolling Sharpe (60d)
    roll = rets.rolling(60)
    rs   = (roll.mean()/roll.std()*np.sqrt(252)).dropna()
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"].iloc[-len(rs):],y=rs,
                             name="Rolling Sharpe",line=dict(color="#ba9eff",width=1.5)))
    fig.add_hline(y=1,line_color="#10B981",line_dash="dot",line_width=1,
                  annotation_text="Good (1.0)",annotation_font_color="#10B981",annotation_font_size=10)
    fig.add_hline(y=0,line_color="#ff6e84",line_dash="dot",line_width=1)
    fig.update_layout(title=dict(text="Rolling Sharpe (60-day)",font=dict(size=13,color="#ecedf6", family="Space Grotesk")),**CT)
    st.plotly_chart(fig, use_container_width=True)


# ── Correlation ────────────────────────────────────────
def tab_correlation():
    data = st.session_state.data
    if len(data) < 2:
        st.info("Load ≥2 assets to see correlation."); return
    closes = {t: obj["df"].set_index("Date")["Close"] for t, obj in data.items()}
    price_df = pd.DataFrame(closes).dropna()
    ret_df   = price_df.pct_change().dropna()
    corr     = ret_df.corr()

    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu", zmin=-1, zmax=1,
                    title="Return Correlation Matrix")
    fig.update_layout(title=dict(font=dict(size=13,color="#ecedf6", family="Space Grotesk")),**CT); st.plotly_chart(fig, use_container_width=True)

    norm = price_df / price_df.iloc[0] * 100
    fig2 = go.Figure()
    pal  = ["#0EA5E9","#f59e0b","#ba9eff","#10B981","#ff6e84","#fbbf24","#34d399"]
    for i,col in enumerate(norm.columns):
        fig2.add_trace(go.Scatter(x=norm.index,y=norm[col],name=col,
                                   line=dict(color=pal[i%len(pal)],width=1.5)))
    fig2.update_layout(title=dict(text="Normalized Performance (Base=100)",
                                   font=dict(size=13,color="#ecedf6", family="Space Grotesk")),**CT)
    st.plotly_chart(fig2, use_container_width=True)


# ── Sentiment ──────────────────────────────────────────
def tab_sentiment():
    t1,t2 = st.tabs(["Manual Text","News Pulse"])
    with t1:
        txt = st.text_area("Paste a headline or paragraph",height=100,
                           placeholder="e.g. Bitcoin smashes $100K as institutional inflows surge…",
                           label_visibility="collapsed")
        if st.button("Analyze Sentiment", type="primary"):
            if txt.strip():
                r = analyze(txt)
                st.session_state.last_sentiment = {"polarity": r.get("polarity", 0.0), "subjectivity": r.get("subjectivity", 0.0)}
                c1,c2,c3 = st.columns(3)
                c1.metric("Sentiment",    f"{r['label']} {r['emoji']}")
                c2.metric("Polarity",     f"{r['polarity']:+.3f}")
                c3.metric("Subjectivity", f"{r['subjectivity']:.3f}")
                color = "#10B981" if r["polarity"]>0.1 else "#ff6e84" if r["polarity"]<-0.1 else "#f59e0b"
                _bar(int((r["polarity"]+1)/2*100), color)
            else:
                st.warning("Enter text first.")
    with t2:
        q = st.text_input("Search topic","Bitcoin",label_visibility="collapsed")
        if st.button("Fetch & Analyze", type="primary"):
            r = news_sentiment(q)
            st.session_state.last_sentiment = {"polarity": r.get("polarity", 0.0), "subjectivity": 0.5}
            c1,c2,c3 = st.columns(3)
            c1.metric("Market Mood",    f"{r['label']} {r['emoji']}")
            c2.metric("Avg Polarity",   f"{r['polarity']:+.3f}")
            c3.metric("Headlines Analyzed",str(r["n"]))
            
            if not r["df"].empty:
                st.markdown('<p style="color:#ecedf6;font-size:1rem;margin:10px 0;font-family:Space Grotesk">📈 Sentiment Trend</p>', unsafe_allow_html=True)
                # Create a simple trend of polarity across the fetched headlines
                fig_s = px.line(r["df"], x=r["df"].index, y="polarity", 
                                title="Pulse Trend", markers=True)
                fig_s.update_traces(line_color="#0EA5E9", marker=dict(size=6))
                fig_s.update_layout(**CT)
                st.plotly_chart(fig_s, use_container_width=True)

            with st.expander("📋 Detailed Headlines"):
                st.dataframe(r["df"], use_container_width=True, hide_index=True)


# ── Live ───────────────────────────────────────────────
def tab_live(ticker, curr):
    price, pct = fetch_live_price(ticker)
    if price is not None:
        clr   = "#22c55e" if pct>=0 else "#ef4444"
        arrow = "▲" if pct>=0 else "▼"
        st.markdown(f"""
        <div style="text-align:center;padding:44px 20px;background:#060c1a;
             border-radius:16px;border:1px solid #0d1f3c;margin:8px 0;
             box-shadow:0 0 40px rgba(56,189,248,.05)">
          <div style="font-family:'JetBrains Mono',monospace;font-size:3.2rem;
               font-weight:700;color:#e2eaf4;letter-spacing:-1px">
            {curr}{price:,.4f}</div>
          <div style="font-size:1.5rem;color:{clr};margin-top:10px;font-weight:700">
            {arrow} {abs(pct):.2f}% today</div>
          <div style="color:#0d2040;font-size:.8rem;margin-top:14px">
            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} local</div>
        </div>""", unsafe_allow_html=True)

        # Check alerts
        for a in st.session_state.price_alerts:
            if a["ticker"]==ticker:
                if (a["type"]=="above" and price>=a["target"]) or \
                   (a["type"]=="below" and price<=a["target"]):
                    st.warning(f"🔔 {ticker} {a['type']} {curr}{a['target']:,.2f}!")
    else:
        st.warning("Live data unavailable. Try refreshing.")

    if st.button("🔄 Refresh", use_container_width=True): st.rerun()

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
    st.markdown("**🔔 Price Alert**")
    ca,cb,cc = st.columns(3)
    al_t = ca.selectbox("Alert type", ["above","below"], label_visibility="collapsed")
    al_v = cb.number_input("Target price", value=float(price or 0),
                            step=0.01, label_visibility="collapsed")
    if cc.button("Set Alert", use_container_width=True):
        st.session_state.price_alerts.append({"ticker":ticker,"type":al_t,"target":al_v})
        st.success(f"Alert: {ticker} {al_t} {curr}{al_v:,.2f}")
    if st.session_state.price_alerts:
        for i,a in enumerate(st.session_state.price_alerts):
            cl,cr = st.columns([4,1])
            cl.markdown(f'<span style="color:#4a6080;font-size:.83rem">{a["ticker"]} {a["type"]} {curr}{a["target"]:,.2f}</span>',
                        unsafe_allow_html=True)
            if cr.button("✕",key=f"da_{i}"):
                st.session_state.price_alerts.pop(i); st.rerun()


# ── Portfolio ──────────────────────────────────────────
def tab_portfolio(curr):
    port = st.session_state.portfolio
    with st.expander("➕ Add / Update Position", expanded=not port.positions):
        with st.form("portfolio_add_form", clear_on_submit=True):
            c1,c2,c3,c4 = st.columns(4)
            pt = c1.text_input("Ticker", placeholder="BTC-USD", key="pf_add_ticker")
            pn = c2.text_input("Name", placeholder="Bitcoin", key="pf_add_name")
            pu = c3.number_input("Units", min_value=0.0001, value=1.0, step=0.001, format="%.4f", key="pf_add_units")
            pp = c4.number_input("Avg Buy", min_value=0.01, value=100.0, step=0.01, key="pf_add_avg")
            submitted = st.form_submit_button("Add Position", type="primary", use_container_width=True)

        if submitted:
            ticker_clean = (pt or "").strip().upper()
            if not ticker_clean:
                st.warning("Please enter a valid ticker symbol.")
            else:
                try:
                    name_clean = (pn or "").strip() or ticker_clean
                    port.add(Position(ticker_clean, name_clean, float(pu), float(pp), currency(ticker_clean)))
                    st.session_state.portfolio_ai_cache = {}
                    st.success(f"✅ Added {pu} × {ticker_clean}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not add position: {e}")
    if not port.positions:
        st.info("Add positions to start tracking."); return

    with st.spinner("Fetching live prices…"):
        live = {t:(fetch_live_price(t)[0] or 0) for t in port.positions}

    s = port.summary(live)
    if not s.empty:
        st.dataframe(s, use_container_width=True, hide_index=True)
        st.markdown('<hr class="sec-div">', unsafe_allow_html=True)
        tv = port.total_value(live); tc = port.total_cost()
        tp = port.total_pnl(live); tpp = tp/tc*100 if tc else 0
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Invested",      f"{curr}{tc:,.2f}")
        c2.metric("Current Value", f"{curr}{tv:,.2f}")
        c3.metric("Total P&L",     f"{curr}{tp:+,.2f}",f"{tpp:+.2f}%")
        c4.metric("Positions",     str(len(port.positions)))
        if live:
            labels = list(live.keys())
            vals   = [port.positions[t].value(live[t]) for t in labels if live[t]]
            if vals:
                fig = px.pie(names=labels,values=vals,title="Portfolio Allocation",
                             color_discrete_sequence=["#38bdf8","#f59e0b","#a78bfa",
                                                       "#4ade80","#f87171","#fbbf24"])
                fig.update_traces(textposition="inside",textinfo="percent+label")
                fig.update_layout(**CT); st.plotly_chart(fig, use_container_width=True)

    rm = st.selectbox("Remove", ["—"]+list(port.positions.keys()))
    if rm!="—" and st.button("🗑️ Remove", key="pf_remove_btn"):
        port.remove(rm)
        st.session_state.portfolio_ai_cache = {}
        st.rerun()


def tab_portfolio_intelligence(curr, horizon):
    port = st.session_state.portfolio
    if not port.positions:
        st.info("Add positions in the Portfolio tab to run AI Portfolio Analyzer.")
        return

    with st.container():
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### Portfolio Intelligence Engine")
            st.caption("Health scoring, diversification diagnostics, risk analytics, model-driven forecasting, and AI rebalancing.")
        with c2:
            run_ai = st.button("▶ Run Portfolio AI Analysis", type="primary", use_container_width=True)

    signature = tuple(sorted((t, float(p.shares), float(p.avg_price)) for t, p in port.positions.items()))
    cache_key = f"{signature}|{horizon}"
    has_cache = cache_key in st.session_state.portfolio_ai_cache
    if not run_ai and not has_cache:
        st.info("Run analysis to generate portfolio health, diversification, forecasting, and scenario intelligence.")
        return

    if run_ai or not has_cache:
        with st.spinner("Building portfolio intelligence using forecasting, GARCH, XGBoost, risk, and backtest engines…"):
            live = {t: (fetch_live_price(t)[0] or 0.0) for t in port.positions}
            history = {}
            for t in port.positions:
                if t in st.session_state.data and not st.session_state.data[t]["df"].empty:
                    history[t] = st.session_state.data[t]["df"]
                else:
                    history[t] = load_asset(t, "1y", "1d")
            st.session_state.portfolio_ai_cache[cache_key] = analyze_portfolio(port, live, history, horizon=horizon)

    result = st.session_state.portfolio_ai_cache[cache_key]
    health = result.get("health")
    if health is None:
        st.warning("Insufficient portfolio data to compute intelligence insights.")
        return

    st.markdown('<hr class="sec-div">', unsafe_allow_html=True)

    # 1) Portfolio Health Score
    left, right = st.columns([1, 2])
    with left:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health.score,
            title={"text": "Portfolio Health"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#38bdf8"},
                "steps": [
                    {"range": [0, 45], "color": "rgba(255,110,132,0.25)"},
                    {"range": [45, 70], "color": "rgba(245,158,11,0.25)"},
                    {"range": [70, 100], "color": "rgba(16,185,129,0.25)"},
                ],
            },
        ))
        gauge.update_layout(height=290, **CT)
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown(f"**Health Label:** {health.health_label}")
        st.markdown(f"**Risk Label:** {health.risk_label}")

    with right:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Diversification", f"{health.components['Diversification']:.1f}")
        c2.metric("Concentration", f"{health.components['Concentration']:.1f}")
        c3.metric("Volatility", f"{health.components['Volatility']:.1f}")
        c4.metric("Sharpe", f"{health.components['Sharpe']:.1f}")
        c5.metric("Sector", f"{health.components['Sector']:.1f}")
        st.markdown("**AI Suggestions**")
        for s in health.suggestions:
            st.markdown(f"- {s}")

    # 2) Asset Allocation Analysis
    st.markdown("### Asset Allocation Analysis")
    a1, a2 = st.columns(2)
    alloc = result["allocation"]
    sectors = result["sectors"]
    with a1:
        if not alloc.empty:
            fig = px.pie(alloc, names="Ticker", values="Value", title="Allocation Breakdown",
                         color_discrete_sequence=["#38bdf8", "#f59e0b", "#a78bfa", "#4ade80", "#f87171", "#fbbf24"])
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(**CT)
            st.plotly_chart(fig, use_container_width=True)
    with a2:
        if not sectors.empty:
            fig = px.pie(sectors, names="Sector", values="Weight", title="Sector Weights",
                         color_discrete_sequence=["#0EA5E9", "#10B981", "#f59e0b", "#ff6e84", "#ba9eff", "#fbbf24"])
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(**CT)
            st.plotly_chart(fig, use_container_width=True)

    # 3) Diversification Analyzer
    st.markdown("### Diversification Analyzer")
    corr = result.get("correlation", pd.DataFrame())
    pairs = result.get("correlated_pairs", [])
    if pairs:
        for a, b, c in pairs[:5]:
            st.warning(f"High correlation detected: {a} vs {b} ({c:.2f})")
    max_w = float(alloc["Weight"].max()) if not alloc.empty else 0.0
    if max_w > 0.35:
        st.warning(f"Concentration warning: largest position is {max_w*100:.1f}% of portfolio.")
    if not corr.empty:
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                        color_continuous_scale="RdBu", title="Holdings Correlation Matrix")
        fig.update_layout(**CT)
        st.plotly_chart(fig, use_container_width=True)

    # 4) Portfolio Risk Dashboard
    st.markdown("### Portfolio Risk Dashboard")
    r = result.get("risk", {})
    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
    rc1.metric("Portfolio Sharpe", f"{r.get('Sharpe', np.nan):.3f}" if pd.notna(r.get("Sharpe", np.nan)) else "N/A")
    rc2.metric("Portfolio Sortino", f"{r.get('Sortino', np.nan):.3f}" if pd.notna(r.get("Sortino", np.nan)) else "N/A")
    rc3.metric("VaR 95%", f"{(r.get('VaR95', np.nan)*100):.2f}%" if pd.notna(r.get("VaR95", np.nan)) else "N/A")
    rc4.metric("Max Drawdown", f"{(r.get('Max Drawdown', np.nan)*100):.2f}%" if pd.notna(r.get("Max Drawdown", np.nan)) else "N/A")
    rc5.metric("Portfolio Volatility", f"{(r.get('Ann Vol', np.nan)*100):.2f}%" if pd.notna(r.get("Ann Vol", np.nan)) else "N/A")

    # 5) Portfolio Forecasting
    st.markdown("### Portfolio Forecasting")
    projection = result.get("projection", pd.DataFrame())
    exp_ret = result.get("portfolio_expected_return_pct", np.nan)
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Forecast Return", f"{exp_ret:+.2f}%" if pd.notna(exp_ret) else "N/A")
    if not projection.empty:
        best_case = (projection["Best"].iloc[-1] / projection["Base"].iloc[0] - 1) * 100
        worst_case = (projection["Worst"].iloc[-1] / projection["Base"].iloc[0] - 1) * 100
        fc2.metric("Best Case", f"{best_case:+.2f}%")
        fc3.metric("Worst Case", f"{worst_case:+.2f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=projection["Date"], y=projection["Base"], name="Base", line=dict(color="#38bdf8", width=2)))
        fig.add_trace(go.Scatter(x=projection["Date"], y=projection["Best"], name="Best", line=dict(color="#10B981", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=projection["Date"], y=projection["Worst"], name="Worst", line=dict(color="#ff6e84", width=1.5, dash="dot")))
        fig.update_layout(title=dict(text="Portfolio Value Projection", font=dict(size=13, color="#ecedf6", family="Space Grotesk")),
                          yaxis_title=f"Portfolio Value ({curr})", **CT)
        st.plotly_chart(fig, use_container_width=True)

    asset_intel = result.get("asset_intel", pd.DataFrame())
    if not asset_intel.empty:
        with st.expander("Model Intelligence by Asset"):
            st.dataframe(asset_intel, use_container_width=True, hide_index=True)

    # 6) Rebalancing Suggestions
    st.markdown("### Rebalancing Suggestions")
    reb = result.get("rebalancing", pd.DataFrame())
    if not reb.empty:
        st.dataframe(reb[["Ticker", "Weight", "Target Weight", "Delta Weight", "Action"]], use_container_width=True, hide_index=True)

    # 7) What-if Scenario Simulator
    st.markdown("### What-if Scenario Simulator")
    t1, t2, t3 = st.tabs(["SIP", "Market Drop Stress", "Asset Impact"])
    with t1:
        s1, s2, s3 = st.columns(3)
        monthly = s1.number_input("Monthly SIP", min_value=0.0, value=1000.0, step=100.0)
        years = s2.slider("Years", 1, 20, 5)
        ann = s3.slider("Expected Annual Return (%)", 1.0, 30.0, 12.0)
        initial_value = float(result.get("portfolio_value", 0.0))
        months = int(years * 12)
        total_contributed = float(monthly) * months
        total_invested = initial_value + total_contributed
        sip_df = simulate_sip(initial_value, monthly, years, ann / 100.0)
        if not sip_df.empty:
            final_value = float(sip_df["Portfolio Value"].iloc[-1])
            net_gain = final_value - total_invested
            return_pct = (net_gain / total_invested * 100) if total_invested > 0 else 0.0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Invested", f"{curr}{total_invested:,.2f}")
            m2.metric("Final Value", f"{curr}{final_value:,.2f}")
            m3.metric("Net Gain", f"{curr}{net_gain:+,.2f}")
            m4.metric("Return", f"{return_pct:+.2f}%")

            with st.expander("SIP Breakdown"):
                b1, b2 = st.columns(2)
                b1.metric("Current Portfolio Base", f"{curr}{initial_value:,.2f}")
                b2.metric("Future SIP Contributions", f"{curr}{total_contributed:,.2f}")

            fig = go.Figure(go.Scatter(x=sip_df["Month"], y=sip_df["Portfolio Value"], line=dict(color="#10B981", width=2), name="SIP"))
            fig.update_layout(title=dict(text="SIP Growth Simulation", font=dict(size=13, color="#ecedf6", family="Space Grotesk")),
                              xaxis_title="Month", yaxis_title=f"Portfolio Value ({curr})", **CT)
            st.plotly_chart(fig, use_container_width=True)

    with t2:
        drop = st.slider("Simulated Market Drop (%)", 5, 50, 20)
        stress = simulate_market_drop(result["positions"], drop)
        if not stress.empty:
            st.dataframe(stress[["Ticker", "Sector", "Value", "Stress Value", "Impact"]], use_container_width=True, hide_index=True)

    with t3:
        c1, c2 = st.columns(2)
        impact_ticker = c1.text_input("Ticker to add/remove", value="SPY")
        impact_value = c2.number_input("Capital change (+/-)", value=5000.0, step=500.0)
        impact_df = simulate_asset_impact(result["positions"], impact_ticker.upper().strip(), impact_value)
        if not impact_df.empty:
            st.dataframe(impact_df, use_container_width=True, hide_index=True)


# ── Export ─────────────────────────────────────────────
def tab_export(df, feat, ticker):
    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("⬇️ OHLCV CSV",df.to_csv(index=False).encode(),
                           f"{ticker}_ohlcv.csv","text/csv",use_container_width=True)
    with c2:
        st.download_button("⬇️ Indicators CSV",feat.to_csv(index=False).encode(),
                           f"{ticker}_indicators.csv","text/csv",use_container_width=True)
    with c3:
        sig = st.session_state.data.get(ticker,{}).get("sig",{})
        rpt = (f"TradeMind AI Signal Report\n{'='*40}\n"
               f"Asset: {ticker}\nDate: {datetime.now()}\n\n"
               f"Signal: {sig.get('signal','N/A')} {sig.get('emoji','')}\n"
               f"Confidence: {sig.get('confidence','N/A')}%\n"
               f"Tech Score: {sig.get('tech_score','N/A')}\n"
               f"Final Score: {sig.get('final_score','N/A')}\n\n"
               f"WHY:\n" + "\n".join(f"  • {w}" for w in sig.get("why",[])) +
               "\n\nRISKS:\n" + "\n".join(f"  • {r}" for r in sig.get("risks",[])))
        st.download_button("⬇️ Signal Report",rpt.encode(),
                           f"{ticker}_signal.txt","text/plain",use_container_width=True)
    st.dataframe(feat.tail(10), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    st.markdown("""
    <div class="tm-hero">
      <h1>🧠 TradeMind AI</h1>
      <p>AI-powered trading intelligence · Crypto · US Stocks · Nifty 50 · Forex · Backtesting · Portfolio</p>
    </div>""", unsafe_allow_html=True)

    _, period, interval, horizon = sidebar()

    if not st.session_state.data:
        # Landing page
        st.markdown("""
        <div style="text-align:center;padding:60px 20px 40px">
          <div style="font-size:5rem;margin-bottom:16px">📊</div>
          <div style="color:#c9d3e0;font-size:1.5rem;font-weight:800;margin-bottom:8px;letter-spacing:-.5px">
            Professional AI Trading Intelligence</div>
          <div style="color:#1e3a5e;font-size:.95rem;max-width:520px;margin:0 auto;line-height:1.7">
            Select assets from the sidebar and click <strong style="color:#38bdf8">Load & Analyze</strong>
            to instantly get AI signals, forecasts, risk metrics, backtests, and portfolio tracking.
          </div>
        </div>
        <div class="feat-grid">
          <div class="feat-card"><div class="feat-icon">🔮</div>
            <div class="feat-title">AI Forecasting</div>
            <div class="feat-desc">ARIMA · Prophet · LSTM with normalized RMSE/MAE metrics and confidence bands</div>
          </div>
          <div class="feat-card"><div class="feat-icon">⚡</div>
            <div class="feat-title">WHY Engine</div>
            <div class="feat-desc">BUY/SELL/HOLD with human-readable reasoning, risk warnings, and multi-factor scoring</div>
          </div>
          <div class="feat-card"><div class="feat-icon">📉</div>
            <div class="feat-title">Backtesting</div>
            <div class="feat-desc">5 strategies · equity curve vs buy-and-hold · win rate · Sharpe · profit factor</div>
          </div>
          <div class="feat-card"><div class="feat-icon">📐</div>
            <div class="feat-title">Correlation</div>
            <div class="feat-desc">Return correlation matrix · normalized performance · multi-asset comparison</div>
          </div>
        </div>""", unsafe_allow_html=True)
        return

    signal_banner()

    ticker = st.selectbox("Analyze →", list(st.session_state.data.keys()),
                           label_visibility="collapsed")
    df   = st.session_state.data[ticker]["df"]
    feat = st.session_state.data[ticker]["feat"]
    curr = currency(ticker)

    tabs = st.tabs([
        "🏠 Overview", "📄 Raw Data",   "📊 Indicators", "🕯️ Patterns",
        "🔮 Forecast",  "🧪 Forecast Accuracy", "📉 Backtest", "🤖 RL Agent Lab", "📈 Risk",        "📐 Correlation",
        "🧠 Sentiment", "📺 Live Price", "💼 Portfolio", "💼 Portfolio AI", "📥 Export",
    ])

    with tabs[0]:  tab_overview(df, feat, ticker, curr)
    with tabs[1]:  tab_raw(df, ticker, curr)
    with tabs[2]:  tab_indicators(feat)
    with tabs[3]:  tab_patterns(feat)
    with tabs[4]:  tab_forecast(df, feat, ticker, horizon)
    with tabs[5]:  tab_forecast_accuracy(df, ticker)
    with tabs[6]:  tab_backtest(feat, ticker, curr)
    with tabs[7]:  tab_rl_agent_lab(df, feat, ticker, curr, horizon)
    with tabs[8]:  tab_risk(df, ticker)
    with tabs[9]:  tab_correlation()
    with tabs[10]: tab_sentiment()
    with tabs[11]: tab_live(ticker, curr)
    with tabs[12]: tab_portfolio(curr)
    with tabs[13]: tab_portfolio_intelligence(curr, horizon)
    with tabs[14]: tab_export(df, feat, ticker)


if __name__ == "__main__":
    main()
