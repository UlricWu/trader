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
    def __init__(self, data, settings: Settings, strategy=Strategy):
        self.events = Queue()

        self.data_handler = DailyBarDataHandler(data=data, events=self.events, settings=settings)
        self.strategy = strategy(self.events, settings=settings)
        self.execution_handler = ExecutionHandler(self.events, settings=settings)
        self.portfolio = Portfolio(self.events, settings=settings)

        if not len(data):
            logs.record_log("策略初始化异常，数据为空", 3)
            return
        logs.record_log("策略初始化完成")

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
            self.portfolio.record_daily_snapshot(event.datetime)

    def plot_equity_curve(self):
        self.portfolio.equity_df.plot(title="Equity Curve", figsize=(10, 5))
        plt.ylabel("Equity")
        plt.show()

    def summary(self):
        if self.strategy and hasattr(self.strategy, "predictions"):
            preds = self.strategy.predictions
            if preds:
                correct = sum(1 for p, a in preds if p == a)
                accuracy = correct / len(preds)
                print(f"ML Prediction Accuracy: {accuracy:.2%} ({correct}/{len(preds)})")

            # if preds:
            #     correct = sum(1 for prob, a in preds if (prob >= 0.5) == a)
            #     accuracy = correct / len(preds)
            #     avg_conf = sum(abs(prob - 0.5) for prob, _ in preds) / len(preds) * 2
            #     print(f"ML Prediction Accuracy: {accuracy:.2%} ({correct}/{len(preds)})")
            #     print(f"Avg Prediction Confidence: {avg_conf:.2%}")
