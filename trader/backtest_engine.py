# backtest_engine.py
from queue import Queue, Empty

from matplotlib import pyplot as plt

from trader.events import EventType
from trader.portfolio import Portfolio
from trader.strategy import Strategy
from trader.execution import SimulatedExecutionHandler
from trader.data_handler import DailyBarDataHandler


class Backtest:
    def __init__(self, data, initial_cash=100000):
        self.events = Queue()
        self.data_handler = DailyBarDataHandler(data=data, events=self.events)

        # Strategy, Execution Handler, and Portfolio
        self.strategy = Strategy(self.events)
        self.execution_handler = SimulatedExecutionHandler(self.events)
        self.portfolio = Portfolio(self.events, initial_cash)

    def run(self):
        """Run the backtest loop."""
        while self.data_handler.continue_backtest:
            self.data_handler.stream_next()

            while not self.events.empty():
                event = self.events.get()

                if event.type == EventType.MARKET:
                    self.strategy.on_market(event)
                    self.portfolio.update_price(event.symbol, event.close)

                elif event.type == EventType.SIGNAL:
                    self.portfolio.on_signal(event)

                elif event.type == EventType.ORDER:
                    # Use close price as market execution price
                    price = self.portfolio.current_prices.get(event.symbol, 0)
                    self.execution_handler.execute_order(event, price)

                elif event.type == EventType.FILL:
                    self.portfolio.on_fill(event)
                else:
                    print(f"backtest Unknown event type: {type(event)}")

        for date, equity in self.portfolio.history:
            print(date, equity)

    def plot_equity_curve(self):
        """Visualize the portfolio equity over time."""
        equity = [equity for _, equity in self.portfolio.history]
        plt.plot(equity)
        plt.title("Equity Curve")
        plt.show()
