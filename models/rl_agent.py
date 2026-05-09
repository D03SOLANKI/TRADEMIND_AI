"""TradeMind AI — Experimental RL Trading Agent (Q-Learning MVP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from features.indicators import add_all
from models import xgboost_model as xgb_model
from models.forecast_validation import walk_forward_validate, simulate_forecast_strategy
from signals.backtester import run_backtest, STRATEGY_NAMES

ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}


def _regime_code(vol: float) -> int:
    if pd.isna(vol):
        return 1
    if vol < 0.012:
        return 0
    if vol < 0.03:
        return 1
    return 2


def _state_bin(value: float, cuts: List[float]) -> int:
    if pd.isna(value):
        return 1
    return int(np.digitize([value], cuts, right=False)[0])


def _build_market_frame(df: pd.DataFrame, sentiment_features: Dict[str, float] | None = None) -> pd.DataFrame:
    data = add_all(df.copy())
    data["Trend"] = (data.get("SMA7", data["Close"]) - data.get("SMA30", data["Close"])) / data["Close"].replace(0, np.nan)
    data["GARCH_Regime"] = data.get("Vol30", pd.Series(np.nan, index=data.index)).apply(_regime_code)

    sentiment_features = sentiment_features or {}
    data["polarity"] = float(sentiment_features.get("polarity", 0.0))
    data["subjectivity"] = float(sentiment_features.get("subjectivity", 0.0))

    # Use XGBoost model as a state feature source.
    try:
        X, y, valid = xgb_model._prepare_dataset(data, sentiment_features=sentiment_features)  # intentional private reuse
        model, scaler = xgb_model._train_classifier(X, y)  # intentional private reuse
        probs = model.predict_proba(scaler.transform(X))[:, 1]
        xgb_series = pd.Series(probs, index=X.index)
        data["XGB_Prob"] = xgb_series.reindex(data.index).ffill().bfill().fillna(0.5)
    except Exception:
        data["XGB_Prob"] = 0.5

    cols = ["Date", "Close", "RSI", "MACD_hist", "BB_pct", "Vol30", "Trend", "XGB_Prob", "GARCH_Regime"]
    out = data[[c for c in cols if c in data.columns]].copy()
    out = out.dropna(subset=["Close"]).reset_index(drop=True)
    return out


def _encode_state(row: pd.Series, position: int) -> Tuple[int, int, int, int, int, int, int, int]:
    return (
        _state_bin(float(row.get("RSI", np.nan)), [30, 70]),
        _state_bin(float(row.get("MACD_hist", np.nan)), [-0.001, 0.001]),
        _state_bin(float(row.get("BB_pct", np.nan)), [0.2, 0.8]),
        _state_bin(float(row.get("Vol30", np.nan)), [0.012, 0.03]),
        _state_bin(float(row.get("Trend", np.nan)), [-0.01, 0.01]),
        _state_bin(float(row.get("XGB_Prob", np.nan)), [0.4, 0.6]),
        int(row.get("GARCH_Regime", 1)),
        int(position),
    )


def _simulate_policy(
    market: pd.DataFrame,
    q: Dict[Tuple, np.ndarray],
    gamma: float,
    trade_penalty: float,
    drawdown_penalty: float,
) -> Dict[str, Any]:
    if len(market) < 5:
        return {
            "actions": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "metrics": {},
            "signal_source": {"action": "HOLD", "confidence": 50.0},
        }

    pos = 0
    equity = 1.0
    peak = 1.0
    rewards = []
    records = []
    trade_entries = []
    open_trade = None

    close = market["Close"].astype(float).values
    for t in range(len(market) - 1):
        row = market.iloc[t]
        state = _encode_state(row, pos)
        qvals = q.get(state, np.zeros(3, dtype=float))
        action = int(np.argmax(qvals))

        trade_cost = 0.0
        if action == 1:  # BUY
            if pos == 0:
                pos = 1
                trade_cost = trade_penalty
                open_trade = {"Entry Date": row["Date"], "Entry": close[t]}
            else:
                trade_cost = trade_penalty * 0.5
        elif action == 2:  # SELL
            if pos == 1:
                pos = 0
                trade_cost = trade_penalty
                if open_trade is not None:
                    ret_trade = (close[t] / open_trade["Entry"] - 1) * 100
                    trade_entries.append(
                        {
                            "Entry Date": open_trade["Entry Date"],
                            "Exit Date": row["Date"],
                            "Entry": open_trade["Entry"],
                            "Exit": close[t],
                            "PnL %": ret_trade,
                            "Result": "✅" if ret_trade > 0 else "❌",
                        }
                    )
                    open_trade = None
            else:
                trade_cost = trade_penalty * 0.5

        ret_next = close[t + 1] / close[t] - 1 if close[t] else 0.0
        pnl = pos * ret_next
        equity *= (1 + pnl)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0

        reward = pnl - trade_cost - drawdown_penalty * dd
        rewards.append(reward)

        records.append(
            {
                "Date": row["Date"],
                "Close": close[t],
                "Action": ACTIONS[action],
                "Position": pos,
                "Reward": reward,
                "Equity": equity,
                "XGB_Prob": row.get("XGB_Prob", np.nan),
                "GARCH_Regime": row.get("GARCH_Regime", np.nan),
            }
        )

    actions = pd.DataFrame(records)
    eq = pd.DataFrame({"Date": actions["Date"], "RL Equity": actions["Equity"]}) if not actions.empty else pd.DataFrame()

    rets = pd.Series(rewards, dtype=float)
    ann = np.sqrt(252)
    sharpe = (rets.mean() / rets.std() * ann) if len(rets) > 2 and rets.std() > 1e-12 else np.nan
    mdd = ((eq["RL Equity"] / eq["RL Equity"].cummax()) - 1).min() if not eq.empty else np.nan

    trades = pd.DataFrame(trade_entries)
    win_rate = float((trades["PnL %"] > 0).mean() * 100) if not trades.empty else 0.0

    # Confidence from Q-spread on latest state
    last_state = _encode_state(market.iloc[-2], int(actions["Position"].iloc[-1]) if not actions.empty else 0)
    q_last = q.get(last_state, np.zeros(3, dtype=float))
    spread = float(np.max(q_last) - np.min(q_last))
    conf = float(max(35.0, min(95.0, 50.0 + spread * 100.0)))
    action_now = ACTIONS[int(np.argmax(q_last))]

    metrics = {
        "Total Return %": round((equity - 1) * 100, 2),
        "Sharpe": round(float(sharpe), 3) if not np.isnan(sharpe) else np.nan,
        "Max Drawdown %": round(float(mdd) * 100, 2) if not np.isnan(mdd) else np.nan,
        "Win Rate %": round(win_rate, 2),
        "Trades": int(len(trades)),
    }

    return {
        "actions": actions,
        "equity_curve": eq,
        "trades": trades,
        "metrics": metrics,
        "signal_source": {"action": action_now, "confidence": conf},
    }


def _build_comparison(df: pd.DataFrame, feat: pd.DataFrame, rl_metrics: Dict[str, Any], rl_equity: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []

    # RL row
    rows.append(
        {
            "Strategy": "RL Agent (Q-Learning)",
            "Return %": rl_metrics.get("Total Return %", np.nan),
            "Sharpe": rl_metrics.get("Sharpe", np.nan),
            "Max DD %": rl_metrics.get("Max Drawdown %", np.nan),
            "Win Rate %": rl_metrics.get("Win Rate %", np.nan),
        }
    )

    # Buy & hold
    close = df["Close"].astype(float)
    bh_ret = (close.iloc[-1] / close.iloc[0] - 1) * 100 if len(close) > 1 and close.iloc[0] != 0 else np.nan
    bh_rets = close.pct_change().dropna()
    bh_sharpe = (bh_rets.mean() / bh_rets.std() * np.sqrt(252)) if len(bh_rets) > 2 and bh_rets.std() > 1e-12 else np.nan
    bh_eq = pd.DataFrame({"Date": df["Date"], "Buy & Hold": close / close.iloc[0]}) if len(close) > 0 and close.iloc[0] != 0 else pd.DataFrame()
    bh_dd = ((bh_eq["Buy & Hold"] / bh_eq["Buy & Hold"].cummax()) - 1).min() * 100 if not bh_eq.empty else np.nan
    rows.append({"Strategy": "Buy & Hold", "Return %": round(float(bh_ret), 2) if pd.notna(bh_ret) else np.nan,
                 "Sharpe": round(float(bh_sharpe), 3) if pd.notna(bh_sharpe) else np.nan,
                 "Max DD %": round(float(bh_dd), 2) if pd.notna(bh_dd) else np.nan,
                 "Win Rate %": np.nan})

    # Existing indicator strategies (top two by return)
    bt_candidates = []
    for name in STRATEGY_NAMES:
        try:
            bt = run_backtest(feat, name, 10000.0)
            m = bt["metrics"]
            row = {
                "Strategy": name,
                "Return %": float(m.get("Total Return %", np.nan)),
                "Sharpe": float(m.get("Sharpe Ratio", np.nan)),
                "Max DD %": float(m.get("Max Drawdown %", np.nan)),
                "Win Rate %": float(m.get("Win Rate %", np.nan)),
            }
            bt_candidates.append((row, bt.get("equity_curve", pd.DataFrame())))
        except Exception:
            continue

    bt_candidates = sorted(bt_candidates, key=lambda x: x[0].get("Return %", -1e9), reverse=True)
    equity_plot = rl_equity.copy()
    if not bh_eq.empty:
        equity_plot = equity_plot.merge(bh_eq, on="Date", how="left") if not equity_plot.empty else bh_eq.copy()

    for row, eq_df in bt_candidates[:2]:
        rows.append({k: round(v, 3) if isinstance(v, float) and pd.notna(v) else v for k, v in row.items()})
        if isinstance(eq_df, pd.DataFrame) and not eq_df.empty and "Date" in eq_df.columns and "Equity" in eq_df.columns:
            col = row["Strategy"]
            tmp = eq_df[["Date", "Equity"]].copy()
            base = float(tmp["Equity"].iloc[0]) if float(tmp["Equity"].iloc[0]) != 0 else 1.0
            tmp[col] = tmp["Equity"] / base
            tmp = tmp[["Date", col]]
            equity_plot = equity_plot.merge(tmp, on="Date", how="left") if not equity_plot.empty else tmp

    # Forecast-driven proxy strategy using XGB probability (integrates forecasting/ML source)
    try:
        market = _build_market_frame(df)
        probs = market["XGB_Prob"].fillna(0.5)
        close_s = market["Close"].astype(float)
        sig = (probs > 0.55).astype(int)
        rets = close_s.pct_change().shift(-1).fillna(0.0)
        strat_rets = sig * rets
        eq = (1 + strat_rets).cumprod()
        ret = (eq.iloc[-1] - 1) * 100 if len(eq) else np.nan
        shp = (strat_rets.mean() / strat_rets.std() * np.sqrt(252)) if strat_rets.std() > 1e-12 else np.nan
        mdd = ((eq / eq.cummax()) - 1).min() * 100 if len(eq) else np.nan
        rows.append({"Strategy": "Forecast-driven (XGB prob)", "Return %": round(float(ret), 2),
                     "Sharpe": round(float(shp), 3) if pd.notna(shp) else np.nan,
                     "Max DD %": round(float(mdd), 2) if pd.notna(mdd) else np.nan,
                     "Win Rate %": round(float((strat_rets > 0).mean() * 100), 2)})
        eq_tmp = pd.DataFrame({"Date": market["Date"], "Forecast-driven (XGB prob)": eq.values})
        equity_plot = equity_plot.merge(eq_tmp, on="Date", how="left") if not equity_plot.empty else eq_tmp
    except Exception:
        pass

    # Forecast-driven benchmark using existing walk-forward ARIMA strategy simulation
    try:
        wfv = walk_forward_validate(
            df,
            "ARIMA",
            horizon=7,
            method="expanding",
            step=14,
            min_train_size=max(60, min(120, len(df) // 2)),
            hit_threshold=5.0,
        )
        fstrat = simulate_forecast_strategy(wfv["history"], threshold_pct=2.0, capital=10000.0)
        fm = fstrat.get("metrics", {})
        rows.append(
            {
                "Strategy": "Forecast-driven (ARIMA WFV)",
                "Return %": float(fm.get("Total Return %", np.nan)),
                "Sharpe": float(fm.get("Sharpe", np.nan)),
                "Max DD %": float(fm.get("Max Drawdown %", np.nan)),
                "Win Rate %": float(fm.get("Win Rate %", np.nan)),
            }
        )
        eqf = fstrat.get("equity_curve", pd.DataFrame())
        if isinstance(eqf, pd.DataFrame) and not eqf.empty and "Date" in eqf.columns and "Equity" in eqf.columns:
            eqf = eqf.copy()
            base = float(eqf["Equity"].iloc[0]) if float(eqf["Equity"].iloc[0]) != 0 else 1.0
            eqf["Forecast-driven (ARIMA WFV)"] = eqf["Equity"] / base
            equity_plot = equity_plot.merge(eqf[["Date", "Forecast-driven (ARIMA WFV)"]], on="Date", how="left") if not equity_plot.empty else eqf[["Date", "Forecast-driven (ARIMA WFV)"]]
    except Exception:
        pass

    comparison = pd.DataFrame(rows)
    return comparison, equity_plot


def train_q_agent(
    df: pd.DataFrame,
    episodes: int = 60,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon: float = 0.25,
    epsilon_decay: float = 0.99,
    min_epsilon: float = 0.05,
    trade_penalty: float = 0.0006,
    drawdown_penalty: float = 0.08,
    sentiment_features: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    market = _build_market_frame(df, sentiment_features=sentiment_features)
    if len(market) < 90:
        raise ValueError("Need at least 90 rows for RL training.")

    q: Dict[Tuple, np.ndarray] = {}
    learning = []

    close = market["Close"].astype(float).values

    for ep in range(episodes):
        pos = 0
        equity = 1.0
        peak = 1.0
        total_reward = 0.0

        for t in range(len(market) - 1):
            row = market.iloc[t]
            state = _encode_state(row, pos)
            if state not in q:
                q[state] = np.zeros(3, dtype=float)

            if np.random.rand() < epsilon:
                action = np.random.randint(0, 3)
            else:
                action = int(np.argmax(q[state]))

            new_pos = pos
            trade_cost = 0.0
            if action == 1:
                if pos == 0:
                    new_pos = 1
                    trade_cost = trade_penalty
                else:
                    trade_cost = trade_penalty * 0.5
            elif action == 2:
                if pos == 1:
                    new_pos = 0
                    trade_cost = trade_penalty
                else:
                    trade_cost = trade_penalty * 0.5

            ret_next = close[t + 1] / close[t] - 1 if close[t] else 0.0
            pnl = new_pos * ret_next
            equity *= (1 + pnl)
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0.0

            reward = pnl - trade_cost - drawdown_penalty * dd
            total_reward += reward

            next_state = _encode_state(market.iloc[t + 1], new_pos)
            if next_state not in q:
                q[next_state] = np.zeros(3, dtype=float)

            td_target = reward + gamma * np.max(q[next_state])
            q[state][action] = q[state][action] + alpha * (td_target - q[state][action])

            pos = new_pos

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        learning.append({"Episode": ep + 1, "Total Reward": total_reward, "Epsilon": epsilon})

    policy = _simulate_policy(
        market=market,
        q=q,
        gamma=gamma,
        trade_penalty=trade_penalty,
        drawdown_penalty=drawdown_penalty,
    )

    feat = add_all(df.copy())
    comparison, equity_plot = _build_comparison(df, feat, policy.get("metrics", {}), policy.get("equity_curve", pd.DataFrame()))

    return {
        "market": market,
        "learning_curve": pd.DataFrame(learning),
        "actions": policy["actions"],
        "equity_curve": policy["equity_curve"],
        "trades": policy["trades"],
        "metrics": policy["metrics"],
        "signal_source": policy["signal_source"],
        "comparison": comparison,
        "equity_comparison": equity_plot,
    }
