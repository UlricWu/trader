#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : backtest.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/10 11:22
from __future__ import annotations

# backtest_engine.py
import pandas as pd
from .portfolio import Fill, Portfolio  # Assuming Fill is a class that handles transactions

from typing import List


class BacktestEngine:
    def __init__(self, df: pd.DataFrame,
                 strategy, symbols: List[str],
                 initial_cash: float = 10000,
                 slippage: float = 0.001,
                 commission: float = 0.001):
        self.df = df.sort_values(by='date').copy()
        self.strategy = strategy
        self.symbols = symbols
        self.portfolio = Portfolio(initial_cash=initial_cash)

        self.initial_cash = initial_cash

        self.slippage = slippage
        self.commission = commission

    def run(self):
        """Run the backtest by iterating through each trading day."""

        prices = {}
        self.df = self.strategy.generate_signals(self.df)
        for symbol, group in self.df.groupby("symbol"):
            group = group.sort_values("date").reset_index(drop=True)

            for _, row in group.iterrows():
                signal = row["signal"]
                price = row["close"]
                date = pd.to_datetime(row["date"])

                if signal == 0:
                    continue

                quantity = signal * 100

                # execution_price = close * (1 + slippage_pct)
                #         # price += self.commission * price * signal
                #         # size = self.strategy.sizing(history, symbol, equity)
                prices[symbol] = price

                fill = Fill(symbol=symbol, quantity=quantity, price=price, date=date)

                self.portfolio.update_position(fill)
                self.portfolio.mark_to_market(prices, date)

    def current_value(self) -> float:
        total = self.portfolio.cash
        for symbol, pos in self.portfolio.positions.items():
            latest_price = self.df[self.df["symbol"] == symbol]["close"].iloc[-1]
            total += pos.quantity * latest_price
        return total

    def get_results(self) -> dict:
        return {
            "final_value": round(self.current_value(), 2),
            "cash": round(self.portfolio.cash, 2),
            "positions": self.portfolio.summary()
        }
