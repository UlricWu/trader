# portfolio.py
from collections import defaultdict

import pandas as pd

from collections import defaultdict
from trader.events import OrderEvent, FillEvent, EventType, SignalEvent


class Portfolio:
    def __init__(self, events, initial_cash=100000):
        self.events = events
        self.cash = initial_cash
        self.holdings = defaultdict(int)
        self.current_prices = {}
        self.history = []

    def update_price(self, symbol, price):
        self.current_prices[symbol] = price
        total_equity = self.equity()
        self.history.append((pd.Timestamp.now(), total_equity))

    def on_signal(self, signal: SignalEvent):
        quantity = 10
        price = self.current_prices.get(signal.symbol, 0)

        # Handle BUY or SELL signals with LIMIT price logic

        if signal.signal_type == "BUY" and self.cash >= price * quantity:
            # limit_price = signal.limit_price if signal.limit_price else price * 1.01
            self.events.put(OrderEvent(symbol=signal.symbol, order_type="MKT", quantity=quantity, direction="BUY",
                                       datetime=signal.datetime))
            print(f"buy {quantity} at {signal.datetime}")
        elif signal.signal_type == "SELL" and self.holdings[signal.symbol] >= quantity:
            self.events.put(OrderEvent(signal.symbol, "MKT", quantity, "SELL", signal.datetime))
        else:
            print(f"Unknown signal type: {signal.signal_type}")

    def on_fill(self, fill):
        cost = fill.price * fill.quantity
        if fill.direction == "BUY":
            self.cash -= cost
            self.holdings[fill.symbol] += fill.quantity
        elif fill.direction == "SELL":
            self.cash += cost
            self.holdings[fill.symbol] -= fill.quantity

    def equity(self):
        equity = self.cash
        for symbol, qty in self.holdings.items():
            price = self.current_prices.get(symbol, 0)
            equity += qty * price
        return equity
