#!filepath: trader/backtest_engine.py

from queue import Queue
from typing import Optional

from trader.event_bus import EventBus
from trader.events import Event, EventType, SignalEvent, OrderEvent, FillEvent
from trader.data_handler import DailyBarDataHandler
from trader.strategy import RuleStrategy
from trader.execution import ExecutionHandler
from trader.portfolio import Portfolio
from trader.risk_manager import RiskManager
from trader.config import Settings
from utilts.logs import logs

from trader.analytics.feature_store import FeatureStore
from trader.analytics.engine import AnalyticsEngine


class Backtest:
    """
    Event-driven backtest engine using:
    - Event queue (FIFO)
    - EventBus for routing with priorities
    - RiskManager stage for SIGNAL filtering
    """

    def __init__(self, data, settings: Settings, strategy_class=RuleStrategy):
        self.events: Queue[Event] = Queue()
        self.settings = settings

        # Core components
        self.data_handler = DailyBarDataHandler(
            data=data, events=self.events, settings=settings
        )
        self.feature_store = FeatureStore()
        self.strategy = strategy_class(settings=settings, feature_store=self.feature_store)
        self.execution_handler = ExecutionHandler(settings=settings)
        self.portfolio = Portfolio(settings=settings)
        self.risk = RiskManager(settings=settings, portfolio=self.portfolio)

        # EventBus with emit -> queue.put
        self.bus = EventBus(emit_fn=self.events.put)

        self.analytics = AnalyticsEngine(feature_store=self.feature_store)

        # -----------------------------
        # Subscriptions with priorities
        # FeatureEngineer → FeatureEvent → Strategy → SignalEvent → Portfolio.
        # -----------------------------

        # MARKET →
        self.bus.subscribe(EventType.MARKET, self._on_analytics_feature, priority=5)
        # self.bus.subscribe(EventType.ANALYTICS, self._on_analytics, priority=5)

        self.bus.subscribe(EventType.MARKET, self._on_market_portfolio, priority=10)
        # ANALYTICS

        # FEATURE → ML pipeline
        # todo
        # ML_FEATURE → strategy
        # MARKET: portfolio (update prices) → strategy (generate SIGNAL)
        self.bus.subscribe(EventType.ANALYTICS, self._on_market_strategy, priority=10)

        # SIGNAL: risk (filter) → portfolio (create ORDER)
        self.bus.subscribe(EventType.SIGNAL, self._on_signal_risk, priority=10)
        self.bus.subscribe(EventType.SIGNAL, self._on_signal_portfolio, priority=20)

        # ORDER: execution (create FILL)
        self.bus.subscribe(EventType.ORDER, self._on_order_exec, priority=10)

        # FILL: portfolio updates state
        self.bus.subscribe(EventType.FILL, self._on_fill_portfolio, priority=10)

        self.bus.subscribe(EventType.SNAPSHOT, self._on_snapshot, priority=10)

    # =============================
    # Event processing
    # =============================

    def _process_event(self, event: Event):
        """Push event into bus for routing to handlers."""
        self.bus.publish(event)

    def run(self):
        """Main backtest loop."""
        logs.record_log("Starting backtest...", 1)
        logs.record_log("开始回放历史数据")
        while self.data_handler.continue_backtest:
            # 1. Pump next market bar
            self.data_handler.stream_next()
            # self.data_handler.update_bars()

            # 2. Process event queue until empty
            while not self.events.empty():
                event = self.events.get()
                if event is None:
                    continue
                last_datetime = getattr(event, "datetime", None)
                self._process_event(event)

            # 3. End-of-day snapshot
            if last_datetime:
                self.bus.emit(Event(EventType.SNAPSHOT, last_datetime))

        logs.record_log("Backtest finished.", 1)

    # =============================
    # Adapters: existing components
    # =============================

    def _on_analytics_feature(self, event: Event, bus: EventBus):
        # print("start feature extraction...")
        feature = self.analytics.on_market(event)
        if feature.no_empty():
            bus.emit(feature)

    def _on_market_portfolio(self, event: Event, bus: EventBus):
        """Update portfolio prices from latest market data."""
        # print('start portfolio update...')
        self.portfolio.update_price(event)

    def _on_market_strategy(self, event: Event, bus: EventBus):
        """Strategy reacts to MARKET event and may generate a SIGNAL."""
        signal: Optional[SignalEvent] = self.strategy.on_analytics(event)
        if signal:
            bus.emit(signal)

    def _on_signal_risk(self, event: Event, bus: EventBus):
        """Risk manager decides whether SIGNAL is allowed."""
        decision = self.risk.decide(event)
        if decision is None:
            logs.record_log(f'risk manager detect risk for {event}', 2)
            # Block propagation (portfolio won’t see it)
            return EventBus.CONSUME

    def _on_signal_portfolio(self, event: Event, bus: EventBus):
        """Portfolio converts SIGNAL into ORDER if allowed."""
        # print('start portfolio')
        order: Optional[OrderEvent] = self.portfolio.on_signal(event)
        if order:
            bus.emit(order)

    def _on_order_exec(self, event: OrderEvent, bus: EventBus):
        """Execution handler converts ORDER into FILL."""

        price = self.portfolio.current_prices.get(event.symbol)
        if price is None:
            logs.record_log(f"No market price for {event.symbol}", 3)
            return
        fill: Optional[FillEvent] = self.execution_handler.execute_order(event, price)
        if fill:
            bus.emit(fill)

    def _on_fill_portfolio(self, event: FillEvent, bus: EventBus):
        """Portfolio updates state from FILL event."""
        # print('start fill portfolio')
        self.portfolio.on_fill(event)

    def _on_snapshot(self, event: Event, bus: EventBus):
        self.portfolio.record_daily_snapshot(event.datetime)  # event.data holds datetime
