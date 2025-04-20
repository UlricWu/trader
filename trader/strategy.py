# strategy.py
from events import SignalEvent
class SMAStrategy:
    def __init__(self, events, short=5, long=20):
        self.events = events
        self.short = short
        self.long = long
        self.prices = []
        self.in_position = False

    def on_market(self, event):
        # self.prices.append(event.price)
        if len(self.prices) < self.long:
            return

        short_ma = sum(self.prices[-self.short:]) / self.short
        long_ma = sum(self.prices[-self.long:]) / self.long

        if short_ma > long_ma and not self.in_position:
            self.events.put(SignalEvent(datetime=event.datetime, symbol=event.symbol, signal_type="BUY"))
            self.in_position = True
        elif short_ma < long_ma and self.in_position:
            self.events.put(SignalEvent(datetime=event.datetime, symbol=event.symbol, signal_type="SELL"))
            self.in_position = False
