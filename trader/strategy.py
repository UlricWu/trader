# strategy.py
from trader.events import SignalEvent, EventType
from collections import defaultdict


from utilts.logs import logs
class Strategy(object):
    slippage = 0.01

    def __init__(self, events, window=3):
        self.events = events
        self.prices = defaultdict(list)

        self.window = window

    def on_market(self, event):
        if event.type != EventType.MARKET:
            logs.record_log(f"Skipping {event}", 3)
            return

        # Update the price history for each symbol
        self.prices[event.symbol].append(event.close)

        if len(self.prices[event.symbol]) < self.window:
            # a guard clause that ensures enough data exists before making a decision.
            logs.record_log(f"Skipping {event} because there are price {self.prices} less than {self.window}", 2)
            return

        avg = sum(self.prices[event.symbol][-self.window:]) / self.window
        if event.close > avg:
            # limit_price = event.close * (1 + self.slippage)  # Buy 1% above the close price
            self.events.put(SignalEvent(symbol=event.symbol, datetime=event.datetime, signal_type="BUY"))
        elif event.close < avg:
            # limit_price = event.close * (1 - self.slippage)  # Sell 1% below the close price
            self.events.put(SignalEvent(symbol=event.symbol, datetime=event.datetime, signal_type="SELL"))
        # else:
        #     print(f"Skipping {event} because error")
