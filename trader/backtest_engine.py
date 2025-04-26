# backtest_engine.py
from queue import Queue, Empty

from matplotlib import pyplot as plt

from trader.events import EventType
from trader.portfolio import Portfolio
from trader.strategy import Strategy
from trader.execution import ExecutionHandler
from trader.data_handler import DailyBarDataHandler
from trader.config import Settings
from utilts.logs import logs


class Backtest:
    def __init__(self, data, settings: Settings):
        self.events = Queue()

        self.data_handler = DailyBarDataHandler(data=data, events=self.events, settings=settings)
        self.strategy = Strategy(self.events, settings=settings)
        self.execution_handler = ExecutionHandler(self.events, settings=settings)
        self.portfolio = Portfolio(self.events, settings=settings)

        # Strategy, Execution Handler, and Portfolio
        # self.strategy = get_strategy(self.events)
        # self.strategy = strategy_cls(self.events)  # avoid hardcoding
        logs.record_log("策略初始化完成")
        # self.portfolio = Portfolio(self.events, initial_cash)

    def run(self):
        """Run the backtest loop."""
        while self.data_handler.continue_backtest:

            logs.record_log("开始回放历史数据")
            self.data_handler.stream_next()

            while not self.events.empty():
                event = self.events.get()

                # if len(self.portfolio.history) % 50 == 0:
                #     print(self.portfolio.history)
                # last_equity = self.portfolio.history
                # logs.record_log(f"Equity at step {len(self.portfolio.history)}: {last_equity}")

                if event.type == EventType.MARKET:
                    self.strategy.on_market(event)
                    self.portfolio.update_price(event)

                elif event.type == EventType.SIGNAL:
                    self.portfolio.on_signal(event)

                elif event.type == EventType.ORDER:
                    # Use close price as market execution price
                    price = self.portfolio.current_prices.get(event.symbol)
                    if price is None:
                        logs.record_log(f"No market price available for {event.symbol}", 3)
                        continue
                    self.execution_handler.execute_order(event, price)

                elif event.type == EventType.FILL:
                    self.portfolio.on_fill(event)
                else:
                    message = f"backtest Unknown event type: {type(event)}"
                    logs.record_log(message, 3)

        for date, equity in self.portfolio.history:
            print(date, equity)

    def plot_equity_curve(self):
        self.portfolio.equity_df.plot(title="Equity Curve", figsize=(10, 5))
        plt.ylabel("Equity")
        plt.show()
