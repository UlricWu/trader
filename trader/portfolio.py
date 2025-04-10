#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : portfolio.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/8 23:30

# !portfolio.py

from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime


@dataclass
class Fill:
    symbol: str
    quantity: int  # Positive = Buy, Negative = Sell
    price: float
    timestamp: datetime


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float


@dataclass
class PortfolioSnapshot:
    date: datetime
    total_value: float
    cash: float
    positions: Dict[str, Position]


class Portfolio:
    def __init__(self, initial_cash: float = 100_000.0):
        self.cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        self.history: List[PortfolioSnapshot] = []
        self._current_value = initial_cash

    def update_lists(self, fills: List[Fill]):
        for fill in fills:
            self._apply_fill(fill)

    def update(self, fill: Fill) -> None:
        self._apply_fill(fill)

    def _apply_fill(self, fill: Fill):
        symbol = fill.symbol
        qty = fill.quantity
        price = fill.price
        cost = qty * price

        self.cash -= cost

        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, qty, price)
        else:
            pos = self.positions[symbol]
            new_qty = pos.quantity + qty

            if new_qty == 0:
                del self.positions[symbol]
            else:
                total_cost = pos.quantity * pos.avg_price + cost
                avg_price = total_cost / new_qty
                self.positions[symbol] = Position(symbol, new_qty, avg_price)

    def mark_to_market(self, prices: Dict[str, float], date: datetime):
        """
                recalculate your total portfolio value, even if no trades happen.
        This is how you generate an equity curve, which is critical for performance tracking.
        :param prices:
        :type prices:
        :param date:
        :type date:
        :return:
        :rtype:
        """
        position_value = sum(
            pos.quantity * prices.get(pos.symbol, 0.0)
            for pos in self.positions.values()
        )
        self._current_value = self.cash + position_value
        snapshot = PortfolioSnapshot(
            date=date,
            total_value=self._current_value,
            cash=self.cash,
            positions=self.positions.copy()
        )
        self.history.append(snapshot)

    @property
    def equity_curve(self) -> List[float]:
        return [snap.total_value for snap in self.history]

    @property
    def dates(self) -> List[datetime]:
        return [snap.date for snap in self.history]

    @property
    def current_value(self) -> float:
        return self._current_value
