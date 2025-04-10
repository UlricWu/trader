#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : portfolio.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/8 23:30

# !portfolio.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from datetime import datetime


@dataclass
class Fill:
    """a completed trade (buy or sell), execution of an order"""
    symbol: str
    quantity: int  # Positive = Buy, Negative = Sell
    price: float
    date: datetime


@dataclass
class Position:
    """ownership state of a stock in the portfolio"""
    symbol: str
    quantity: int
    avg_price: float  # The average price at which shares were purchased

    # todo =add total_cost_with_fee
    # broker_commission_rate: float = 0.0003  # 0.03% broker commission
    # transaction_fee_rate: float = 0.0005  # 0.05% transaction fee
    # stamp_duty_rate: float = 0.001  # 0.1% stamp duty (only for sell transactions)

    def __add__(self, other: Union['Position', Fill]) -> "Position":
        if self.symbol != other.symbol:
            raise ValueError(f"Symbol mismatch: {self.symbol} vs {other.symbol}")

        if isinstance(other, Position):
            price = other.avg_price
        elif isinstance(other, Fill):
            price = other.price
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")

        # Calculate the new quantity and average price
        total_cost = self.avg_price * self.quantity + other.quantity * price

        # Apply transaction fees (broker commission and transaction fee)
        # total_cost_with_fee = total_cost * (1 + self.broker_commission_rate + self.transaction_fee_rate)

        total_quantity = self.quantity + other.quantity
        new_avg_price = total_cost / total_quantity if total_quantity != 0 else 0

        # Return a new Position with updated quantity and avg_price
        return Position(symbol=self.symbol, quantity=total_quantity, avg_price=new_avg_price)
        # broker_commission_rate=self.broker_commission_rate,
        # transaction_fee_rate=self.transaction_fee_rate,
        # stamp_duty_rate=self.stamp_duty_rate)


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    positions: Dict[str, int]
    cash: float
    market_value: float
    total_value: float  # cash + market_value (total value of portfolio)

    def __repr__(self):
        return f"PortfolioSnapshot(cash={self.cash}, market_value={self.market_value}, total_value={self.total_value}, timestamp={self.timestamp})"


@dataclass
class TransactionHistory:
    timestamp: datetime
    symbol: str
    quantity: int
    price: float
    transaction_type: str  # 'buy', 'sell', or 'mtm' (for mark-to-market)
    # slippage: Optional[float] = None
    # commission: Optional[float] = None


class Portfolio:
    def __init__(self, initial_cash: float = 100000.0):
        self.cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        self.history: List[TransactionHistory] = []
        self.snapshots = []
        self._current_value = initial_cash

    def update(self, fills):
        # Process trade fills and update portfolio state
        for fill in fills:
            self.update_position(fill)

    def update_position(self, fill: Fill):
        cost = fill.quantity * fill.price

        # Update cash
        self.cash -= cost

        if fill.symbol in self.positions:
            # If the position exists, use __add__ to update it
            self.positions[fill.symbol] += fill
        else:
            # If position does not exist, create it from the fill
            self.positions[fill.symbol] = Position(symbol=fill.symbol, quantity=fill.quantity, avg_price=fill.price)
            # Update position

        # Record this fill in history
        # Record transaction in history as a structured event
        history_event = TransactionHistory(
            timestamp=fill.date,
            symbol=fill.symbol,
            quantity=fill.quantity,
            price=fill.price,
            transaction_type="buy" if fill.quantity > 0 else "sell",
        )
        self.history.append(history_event)

        # After each update, take a snapshot of the portfolio
        self.take_snapshot()

    def market_value(self, current_prices: Dict[str, float] = None) -> float:
        """ Calculate the total market value of the portfolio. """

        if not current_prices:
            return sum(
                pos.quantity * pos.avg_price for symbol, pos in self.positions.items()
            )

        total_value = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_value += position.quantity * current_prices[symbol]  # position quantity * current price

        return total_value

    def take_snapshot(self, prices: Dict[str, float] = None) -> None:
        """ Record a snapshot of the current portfolio state. """
        market_value = self.market_value(prices)
        total_value = self.cash + market_value
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            positions={symbol: position.quantity for symbol, position in self.positions.items()},
            cash=self.cash,
            market_value=market_value,
            total_value=total_value
        )

        self.snapshots.append(snapshot)

    def mark_to_market(self, prices: Dict[str, float], timestamp: datetime) -> None:
        """ Perform a mark-to-market update on portfolio. """
        value = self.cash + self.market_value(prices)

        # Add mark-to-market as a separate event in history
        mtm_event = TransactionHistory(
            timestamp=timestamp,
            symbol="MTM",
            quantity=0,
            price=value,
            transaction_type="mtm"
        )
        self.history.append(mtm_event)

        # After mark-to-market, take a snapshot of the portfolio
        self.take_snapshot()
