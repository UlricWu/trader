# trader/risk_manager.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd

from trader.events import SignalEvent, OrderEvent, EventType, MarketEvent, Event
from trader.portfolio import Portfolio
from trader.config import Settings
from utilts.logs import logs

from enum import Enum, auto
from typing import Optional


@dataclass
class RiskParams:
    max_position_loss: float
    max_portfolio_dd: float
    default_qty: int
    max_position_size: int
    max_equity_seen: float


class Action(Enum):
    HOLD = auto()
    ENTER = auto()
    EXIT = auto()


@dataclass
class Decision:
    action: Action
    target_qty: int
    reasons: list = None


class RiskManager:
    """
    Centralized risk controls between Strategy and Execution.
    - filter_signal: veto signals under trading halt
    - filter_order: final veto/resize orders
    - on_market: emit forced exit orders (e.g., stop-loss) from fresh prices
    """

    def __init__(self, settings: Settings, portfolio: Portfolio):
        # Pull your risk config from Settings (customize names to your config)
        self.settings = settings
        self.portfolio = portfolio

        self.params = RiskParams(
            max_position_loss=getattr(settings.risk, "stop_loss_pct", 0.05),  # 10% stop-loss per trade
            max_portfolio_dd=getattr(settings.risk, "max_drawdown_pct", 0.2),  # 20% max drawdown allowed
            default_qty=getattr(settings.risk, "default_qty", 100),
            max_position_size=getattr(settings.risk, "max_position_size", 100),
            max_equity_seen=portfolio.equity  # track peak equity
        )

    def decide(self, event: SignalEvent) -> Optional[OrderEvent]:
        """
        Take a SignalEvent and decide whether to allow it.
        Returns an OrderEvent if allowed, otherwise None.
        """

        if event.is_empty():
            return  # block

        if event.type not in {EventType.SIGNAL, EventType.ORDER}:
            return  # let non-trading events pass

        # If it's an order, clamp size. To prevent oversized trades.
        if hasattr(event, "quantity"):
            if abs(event.quantity) > self.params.max_position_size:
                event.quantity = self.params.max_position_size * (1 if event.quantity > 0 else -1)

        # 🔹 Check portfolio-level capital protection
        if self._breach_drawdown():
            return  # block trading completely

        # 🔹 Check stop-loss rules
        if event.type == EventType.SIGNAL:
            if not self._check_stop_loss(event.symbol):
                return  # block this trade

        # logs.record_log(f"risk approve Signal {event} ")

        return event

    # -----------------------------
    # Risk Rule Helpers
    # -----------------------------
    def _breach_drawdown(self) -> bool:
        """
        Stop trading if portfolio drawdown exceeds allowed threshold.
        """
        current_equity = self.portfolio.equity
        self.max_equity_seen = max(self.params.max_equity_seen, current_equity)
        dd = 1 - (current_equity / self.params.max_equity_seen)
        return dd >= self.params.max_portfolio_dd

    def _check_stop_loss(self, symbol: str) -> bool:
        """
        Check if symbol position has breached stop-loss.
        """
        if symbol not in self.portfolio.positions:
            return True  # no position yet, safe to trade

        pos = self.portfolio.positions[symbol]
        price = self.portfolio.current_prices.get(symbol, 0.0)
        if pos.quantity <= 0 or price <= 0:
            return True

        # unrealized PnL %
        pnl_pct = (price - pos.avg_price) / pos.avg_price
        if pnl_pct <= -self.params.max_position_loss:
            # Block further buys
            return False
        return True
