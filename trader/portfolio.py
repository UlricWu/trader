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

    @property
    def stats(self):
        return {
            "cash": self.cash,
            "holdings": dict(self.holdings),
            "current_prices": dict(self.current_prices),
            "equity": self.equity
        }

    def update_price(self, symbol, price):
        self.current_prices[symbol] = price
        total_equity = self.equity
        self.history.append((pd.Timestamp.now(), total_equity))

    def on_signal(self, signal: SignalEvent):
        quantity = 10
        price = self.current_prices.get(signal.symbol, 0)

        # Handle BUY or SELL signals with LIMIT price logic

        if signal.signal_type == "BUY":
            if self.cash < price * quantity:
                print("ERROR BUY SIGNAL {} fail at {}".format(signal.symbol, quantity))
                return

            self.events.put(OrderEvent(symbol=signal.symbol, order_type="MKT", quantity=quantity, direction="BUY",
                                       datetime=signal.datetime))

        elif signal.signal_type == "SELL":
            if self.holdings[signal.symbol] < quantity:
                print("ERROR SELL SIGNAL {} fail at {}".format(signal.symbol, quantity))
                return
            self.events.put(OrderEvent(signal.symbol, "MKT", quantity, "SELL", signal.datetime))

        else:
            print(f"Unknown signal type: {signal.signal_type} for {signal.symbol} at {signal.datetime} {self.stats} ")

    def on_fill(self, fill):
        cost = fill.price * fill.quantity
        if fill.direction == "BUY":
            self.cash -= cost
            self.holdings[fill.symbol] += fill.quantity
        elif fill.direction == "SELL":
            self.cash += cost
            self.holdings[fill.symbol] -= fill.quantity

    @property
    def equity(self):
        equity = self.cash
        for symbol, qty in self.holdings.items():
            price = self.current_prices.get(symbol, 0)
            equity += qty * price
        return equity
