# backtest_engine.py
from queue import Queue, Empty

from matplotlib import pyplot as plt

from trader.events import EventType, Event
from trader.portfolio import Portfolio
from trader.rulestrategy import RuleStrategy
from trader.execution import ExecutionHandler
from trader.data_handler import DailyBarDataHandler
from trader.config import Settings
from utilts.logs import logs


class Backtest:
    def __init__(self, data, settings: Settings, strategy=RuleStrategy):
        self.events = Queue()
        self.data_handler = DailyBarDataHandler(data=data, events=self.events, settings=settings)
        self.strategy = strategy(settings=settings)
        self.execution_handler = ExecutionHandler( settings=settings)
        self.portfolio = Portfolio( settings=settings)

    def run(self):
        """
        Execute the backtest loop.
        """
        logs.record_log("Starting backtest loop...", 1)

        #
        while self.data_handler.continue_backtest:

            logs.record_log("开始回放历史数据")
            self.data_handler.stream_next()

            while not self.events.empty():

                event = self.events.get()

                if event.is_empty():
                    print("This event is uninitialized.")

                if event.type == EventType.MARKET:
                    signal = self.strategy.on_market(event)
                    self.portfolio.update_price(event)
                    self.events.put(signal)


                elif event.type == EventType.SIGNAL:
                    signal = self.portfolio.on_signal(event)
                    self.events.put(signal)

                elif event.type == EventType.ORDER:
                    # Use close price as market execution price
                    price = self.portfolio.current_prices.get(event.symbol)
                    if price is None:
                        logs.record_log(f"No market price available for {event.symbol}", 3)
                        continue
                    signal = self.execution_handler.execute_order(event, price)
                    self.events.put(signal)

                elif event.type == EventType.FILL:
                    self.portfolio.on_fill(event)
                else:
                    message = f"backtest Unknown event type: {type(event)}"
                    logs.record_log(message, 3)
            if event is not None:
                self.portfolio.record_daily_snapshot(event.datetime)

    def plot_equity_curve(self):
        self.portfolio.equity_df.plot(title="Equity Curve", figsize=(10, 5))
        plt.ylabel("Equity")
        plt.show()

    #
    def summary(self):
        logs.record_log(f"Avg Prediction Confidence: ")
        # pass
    #     correct = sum(1 for prob, a in preds if (prob >= 0.5) == a)
    #     accuracy = correct / len(preds)
    #     avg_conf = sum(abs(prob - 0.5) for prob, _ in preds) / len(preds) * 2
    #     print(f"ML Prediction Accuracy: {accuracy:.2%} ({correct}/{len(preds)})")
    #     print(f"Avg Prediction Confidence: {avg_conf:.2%}")
