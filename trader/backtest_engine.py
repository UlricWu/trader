# backtest_engine.py
from queue import Queue, Empty

from trader.events import EventType


class BacktestEngine:
    def __init__(self, data_handler, strategy, execution_handler, portfolio):
        self.events = data_handler.events
        self.data_handler = data_handler
        self.strategy = strategy
        self.execution_handler = execution_handler
        self.portfolio = portfolio

    def run(self):
        while self.data_handler.continue_backtest:
            self.data_handler.update_market()

            while not self.events.empty():
                event = self.events.get()

                if event.type == EventType.MARKET:
                    self.strategy.on_market(event)
                    self.portfolio.on_market(event)

                elif event.type == EventType.SIGNAL:
                    self.execution_handler.on_signal(event)

                elif event.type == EventType.ORDER:
                    self.execution_handler.on_order(event, price=self.portfolio.current_price)

                elif event.type == EventType.FILL:
                    self.portfolio.on_fill(event)
