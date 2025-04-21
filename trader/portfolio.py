# portfolio.py
from collections import defaultdict

import pandas as pd

from collections import defaultdict
from trader.events import OrderEvent, FillEvent, EventType, SignalEvent

from utilts.logs import logs


class Portfolio:
    # Commission and fee rates (China stock market)
    commission_rate = 0.0003  # 0.03%
    min_commission = 5.0  # Min ¥5
    stamp_duty_rate = 0.001  # 0.1% (sell only)
    transfer_fee_rate = 0.00001  # 0.001% (sell only)

    def __init__(self, events, initial_cash=100000, Commission: bool = False):
        self.events = events
        self.cash = initial_cash
        self.holdings = defaultdict(int)
        self.current_prices = {}
        self.history = []

        self.Commission = Commission

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
        signal_symbol = signal.symbol
        price = self.current_prices.get(signal_symbol, 0)

        # Handle BUY or SELL signals with LIMIT price logic

        signal_type = signal.signal_type
        if signal_type == "BUY":
            if self.cash < price * quantity:
                message = f"SIGNAL={signal_type}  fail at  {quantity} because of cash {self.cash} < {price * quantity}"
                logs.record_log(message=message, level=3)
                return

            self.events.put(OrderEvent(symbol=signal_symbol, order_type="MKT", quantity=quantity, direction="BUY",
                                       datetime=signal.datetime))

        elif signal_type == "SELL":
            if self.holdings[signal_symbol] < quantity:
                message = f"SIGNAL={signal_type} {signal_symbol} fail at quantity={quantity} because of not enough holdings {self.holdings}"
                logs.record_log(message=message, level=3)
                return
            self.events.put(OrderEvent(signal_symbol, "MKT", quantity, "SELL", signal.datetime))

        else:
            message = f"Unknown signal type: {signal_type} for {signal_symbol} at {signal.datetime} {self.stats} "
            logs.record_log(message=message, level=3)

    def on_fill(self, fill):
        cost = fill.price * fill.quantity
        if fill.direction == "BUY":
            commission = self.calculate_buy_commission(cost) if self.Commission else 0
            self.cash -= cost + commission  # Deduct commission on buy

            self.holdings[fill.symbol] += fill.quantity
        elif fill.direction == "SELL":
            commission = self.calculate_sell_commission(cost) if self.Commission else 0
            self.cash += cost - commission
            self.holdings[fill.symbol] -= fill.quantity

    @property
    def equity(self):
        equity = self.cash
        for symbol, qty in self.holdings.items():
            price = self.current_prices.get(symbol, 0)
            equity += qty * price
        return equity

    def calculate_buy_commission(self, amount):
        commission = max(amount * self.commission_rate, self.min_commission)
        return commission

    def calculate_sell_commission(self, amount):
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_duty = amount * self.stamp_duty_rate
        transfer_fee = amount * self.transfer_fee_rate
        total_fee = commission + stamp_duty + transfer_fee
        return total_fee
