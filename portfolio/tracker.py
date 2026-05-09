"""TradeMind AI — Portfolio Tracker"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional


@dataclass
class Position:
    ticker:    str
    name:      str
    shares:    float
    avg_price: float
    currency:  str = "$"
    added:     str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_price

    def value(self, price: float) -> float:
        return self.shares * price

    def pnl(self, price: float) -> float:
        return self.value(price) - self.cost_basis

    def pnl_pct(self, price: float) -> float:
        return self.pnl(price) / self.cost_basis * 100 if self.cost_basis else 0


class Portfolio:
    def __init__(self):
        self.positions: Dict[str, Position] = {}

    def add(self, pos: Position):
        if pos.ticker in self.positions:
            ex = self.positions[pos.ticker]
            total_shares = ex.shares + pos.shares
            total_cost   = ex.cost_basis + pos.cost_basis
            ex.shares    = total_shares
            ex.avg_price = total_cost / total_shares if total_shares else 0
        else:
            self.positions[pos.ticker] = pos

    def remove(self, ticker: str):
        self.positions.pop(ticker, None)

    def total_cost(self) -> float:
        return sum(p.cost_basis for p in self.positions.values())

    def total_value(self, prices: Dict[str, float]) -> float:
        return sum(p.value(prices[t]) for t, p in self.positions.items() if t in prices)

    def total_pnl(self, prices: Dict[str, float]) -> float:
        return self.total_value(prices) - self.total_cost()

    def summary(self, prices: Dict[str, float]) -> pd.DataFrame:
        rows = []
        for t, p in self.positions.items():
            px = prices.get(t)
            if px is None:
                continue
            rows.append({
                "Ticker":        t,
                "Name":          p.name,
                "Units":         p.shares,
                "Avg Buy":       f"{p.currency}{p.avg_price:,.2f}",
                "Current Price": f"{p.currency}{px:,.2f}",
                "Invested":      f"{p.currency}{p.cost_basis:,.2f}",
                "Value":         f"{p.currency}{p.value(px):,.2f}",
                "P&L":           f"{p.currency}{p.pnl(px):+,.2f}",
                "P&L %":         f"{p.pnl_pct(px):+.2f}%",
                "Dir":           "📈" if p.pnl(px) >= 0 else "📉",
            })
        return pd.DataFrame(rows)
