# portfolio.py
from typing import Dict

import pandas as pd

from collections import defaultdict
from trader.events import OrderEvent, FillEvent, EventType, SignalEvent, MarketEvent

from utilts.logs import logs

from trader.config import Settings
class Portfolio:
    """
    The Portfolio class tracks cash, positions, and market values over time.
    It records per-symbol market values and total portfolio value at each market update,
    enabling performance analysis and plotting versus a benchmark.
    """
    # Commission and fee rates (China stock market)
    commission_rate = 0.0003  # 0.03%
    min_commission = 5.0  # Min ¥5
    stamp_duty_rate = 0.001  # 0.1% (sell only)
    transfer_fee_rate = 0.00001  # 0.001% (sell only)

    def __init__(self, events, settings: Settings):

        self.events = events
        self.settings = settings
        self.cash = settings.trading.INITIAL_CASH
        self.holdings = defaultdict(int)
        self.current_prices = {}
        self.history = []

        # # positions: symbol -> quantity held
        self.positions: Dict[str, float] = {}

        self.Commission = settings.trading.COMMISSION_RATE
        self.risk_pct = settings.trading.RISK_PCT

        self.buy_dates = defaultdict(list)  # symbol -> list of buy dates
        self.current_date = None

    @property
    def stats(self):
        return {
            "cash": self.cash,
            "holdings": dict(self.holdings),
            "current_prices": dict(self.current_prices),
            "equity": self.equity
        }

    def update_price(self, event: MarketEvent):
        self.current_prices[event.symbol] = event.close
        total_equity = self.equity
        self.history.append((event.datetime, total_equity))

    def on_signal(self, signal: SignalEvent):
        signal_symbol = signal.symbol
        price = self.current_prices.get(signal_symbol, 0)
        # quantity = self.calculate_quantity(price)
        quantity = 10

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

    def on_fill(self, event: FillEvent) -> None:
        """
        Process a FillEvent: deduct cash and update position quantities.
        """
        cost = event.price * event.quantity
        if event.direction == "BUY":
            commission = self.calculate_buy_commission(cost) if self.Commission else 0
            self.cash -= cost + commission  # Deduct commission on buy

            self.holdings[event.symbol] += event.quantity

        elif event.direction == "SELL":
            commission = self.calculate_sell_commission(cost) if self.Commission else 0
            self.cash += cost - commission
            self.holdings[event.symbol] -= event.quantity

    @property
    def equity(self):
        equity = self.cash
        for symbol, qty in self.holdings.items():
            price = self.current_prices.get(symbol, 0)
            equity += qty * price
        return equity

    @property
    def equity_df(self):
        return pd.DataFrame(self.history, columns=["date", "equity"]).set_index("datetime")

    def get_symbol_returns(self) -> dict:
        """
        Returns a dict of {symbol: pd.Series of daily returns}
        """
        return {
            sym: pd.Series([r for _, r in rets], index=[dt for dt, _ in rets])
            for sym, rets in self.symbol_returns.items()
        }

    def calculate_quantity(self, price):
        risk_amount = self.cash * self.risk_pct
        quantity = int(risk_amount // price)
        return max(quantity, 1)

    def calculate_buy_commission(self, amount):
        commission = max(amount * self.commission_rate, self.min_commission)
        return commission

    def calculate_sell_commission(self, amount):
        commission = max(amount * self.commission_rate, self.min_commission)
        stamp_duty = amount * self.stamp_duty_rate
        transfer_fee = amount * self.transfer_fee_rate
        total_fee = commission + stamp_duty + transfer_fee
        return total_fee
