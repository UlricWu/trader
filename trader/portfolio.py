# portfolio.py
class Portfolio:
    def __init__(self):
        self.cash = 100000
        self.position = 0
        self.current_price = 0
        self.history = []

    def on_fill(self, fill_event):
        direction = 1 if fill_event.direction == "BUY" else -1
        self.position += direction * fill_event.quantity
        self.cash -= direction * fill_event.quantity * fill_event.price
        self.current_price = fill_event.price

    def on_market(self, market_event):
        self.current_price = market_event.price
        equity = self.cash + self.position * self.current_price
        self.history.append((market_event.datetime, equity))
