# backtest_engine.py
from queue import Queue
from matplotlib import pyplot as plt
from typing import Optional

from trader.events import EventType, Event
from trader.portfolio import Portfolio
from trader.rulestrategy import RuleStrategy
from trader.execution import ExecutionHandler
from trader.data_handler import DailyBarDataHandler
from trader.config import Settings
from utilts.logs import logs


class Backtest:
    """
    Event-driven backtesting engine.
    Manages data streaming, strategy execution, order execution, and portfolio updates.
    """

    def __init__(self, data, settings: Settings, strategy_class=RuleStrategy):
        self.events: Queue[Event] = Queue()
        self.settings = settings

        self.data_handler = DailyBarDataHandler(
            data=data, events=self.events, settings=settings
        )

        # Strategy only generates signals; engine enqueues them
        self.strategy = strategy_class(settings=settings)

        self.execution_handler = ExecutionHandler(settings=settings)
        self.portfolio = Portfolio(settings=settings)

    # -----------------------------
    # Public API
    # -----------------------------
    def run(self):
        """Run the backtest loop."""
        logs.record_log("Starting backtest...", 1)

        while self.data_handler.continue_backtest:
            self.data_handler.stream_next()

            while not self.events.empty():
                event: Optional[Event] = self.events.get()
                if event is None or event.is_empty():
                    continue

                self._process_event(event)

            # Record end-of-day snapshot
            last_datetime = getattr(event, "datetime", None)
            if last_datetime:
                self.portfolio.record_daily_snapshot(last_datetime)

        logs.record_log("Backtest completed.", 1)

    def plot_equity_curve(self):
        """Plot the total portfolio equity over time."""
        self.portfolio.equity_df.plot(title="Equity Curve", figsize=(10, 5))
        plt.ylabel("Equity")
        plt.show()

    # -----------------------------
    # Internal Methods
    # -----------------------------
    def _process_event(self, event: Event):
        """Dispatch event to the appropriate component."""
        if event.type == EventType.MARKET:
            signal_event = self.strategy.on_market(event)
            self.portfolio.update_price(event)
            if signal_event:
                self.events.put(signal_event)

        elif event.type == EventType.SIGNAL:
            order_event = self.portfolio.on_signal(event)
            if order_event:
                self.events.put(order_event)

        elif event.type == EventType.ORDER:
            price = self.portfolio.current_prices.get(event.symbol)
            if price is None:
                logs.record_log(f"No market price available for {event.symbol}", 3)
                return
            fill_event = self.execution_handler.execute_order(event, price)
            if fill_event:
                self.events.put(fill_event)

        elif event.type == EventType.FILL:
            self.portfolio.on_fill(event)

        else:
            logs.record_log(f"Unknown event type: {event.type}", 3)
