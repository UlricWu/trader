#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : backtest.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/10 11:22
from __future__ import annotations

# !backtest.py

from .portfolio import Portfolio, Fill
from datetime import datetime
import pandas as pd
from .strategy import Strategy

# def generate_fill(symbol: str, signal: int, price: float, date: datetime) -> Fill | None:
#     if signal == 1:
#         return Fill(symbol=symbol, quantity=10, price=price, timestamp=date)
#     elif signal == -1:
#         return Fill(symbol=symbol, quantity=-10, price=price, timestamp=date)
#     return None  # 0 Hold — no action taken
#
#
# def run_backtest(df: pd.DataFrame, strategy, symbol="AAPL"):
#     portfolio = Portfolio()
#     features = df.copy()
#     strategy.train(features)
#
#     for date, row in features.iterrows():
#         current_price = row["close"]
#         signal = strategy.predict_signal(features.loc[:date])
#
#         fill = generate_fill(symbol, signal, current_price, date)
#         if fill:  # Buy
#             portfolio.update(fill)
#
#         portfolio.mark_to_market({symbol: current_price}, date)
#
#     return portfolio

# class Backtest:
#     def __init__(
#         self,
#         df: pd.DataFrame,
#         strategy: Strategy,
#         symbols: [str],
#         initial_cash: float = 100_000
#     ):
#         self.df = df
#         self.strategy = strategy
#         self.symbols = symbols
#         self.portfolio = Portfolio(initial_cash=initial_cash)
#
#     def generate_fill(self, symbol: str, signal: int, size: int, price: float, date: datetime) -> Optional[Fill]:
#         if size == 0:
#             return None
#         if signal == 1:
#             return Fill(symbol=symbol, quantity=size, price=price, timestamp=date)
#         elif signal == -1:
#             return Fill(symbol=symbol, quantity=-size, price=price, timestamp=date)
#         return None
#
#     def run(self) -> Portfolio:
#         self.strategy.train(self.df)
#
#         for date in sorted(self.df["date"].unique()):
#             symbol_prices = {}
#             day_data = self.df[self.df["date"] == date]
#
#             for symbol in self.symbols:
#                 row = day_data[day_data["symbol"] == symbol]
#                 if row.empty:
#                     continue
#
#                 current_price = row.iloc[0]["close"]
#                 history = self.df[(self.df["symbol"] == symbol) & (self.df["date"] <= date)]
#
#                 signal = self.strategy.predict_signal(history, symbol)
#                 equity = self.portfolio.cash + self.portfolio.market_value()
#                 size = self.strategy.sizing(history, symbol, equity)
#
#                 fill = self.generate_fill(symbol, signal, size, current_price, date)
#                 if fill:
#                     self.portfolio.update([fill])
#
#                 symbol_prices[symbol] = current_price
#
#             self.portfolio.mark_to_market(symbol_prices, date)
#
#         return self.portfolio
