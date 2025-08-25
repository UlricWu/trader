# execution.py
from __future__ import annotations
from typing import Optional

from trader.events import FillEvent, OrderEvent, Event, EventType
from trader.config import Settings
from utilts.logs import logs


class ExecutionHandler:
    """
    Simulates execution of orders with optional slippage and limit price checks.
    """
    DIRECTION_MAPPING = {
        "BUY": 1,
        "SELL": -1
    }

    def __init__(self, settings: Settings):
        self.settings = settings
        self.slippage_pct: float = settings.trading.SLIPPAGE

    # -----------------------------
    # Public API
    # -----------------------------
    def execute_order(
        self,
        order_event: OrderEvent,
        market_price: float,
        prev_close: Optional[float] = None
    ) -> Event:
        """Execute an OrderEvent and return a FillEvent or skip Event if not executable."""
        if order_event.type != EventType.ORDER:
            logs.record_log(
                f"Invalid event type {order_event.type}, expected ORDER.",
                level=3
            )
            return

        executed_price = self._get_executed_price(order_event, market_price)
        if executed_price is None:
            logs.record_log(f'no meet price for order {order_event}')
            return   # Skip execution

        fill = FillEvent(
            symbol=order_event.symbol,
            price=executed_price,
            quantity=order_event.quantity,
            direction=order_event.direction,
            datetime=order_event.datetime
        )

        logs.record_log(f"Executed {order_event.order_type} order for {order_event.symbol} at {executed_price}")
        return fill

    # -----------------------------
    # Internal Methods
    # -----------------------------
    def _get_executed_price(self, order_event: OrderEvent, market_price: float) -> Optional[float]:
        """Determine executed price based on order type and slippage."""
        if order_event.order_type == "MKT":
            return self._simulate_slippage(market_price, order_event)
        elif order_event.order_type == "LIMIT":
            if order_event.direction == "BUY" and market_price > order_event.limit_price:
                logs.record_log(f"Blocked BUY limit order for {order_event.symbol} at {market_price}", level=2)
                return None
            if order_event.direction == "SELL" and market_price < order_event.limit_price:
                logs.record_log(f"Blocked SELL limit order for {order_event.symbol} at {market_price}", level=2)
                return None
            return order_event.limit_price
        else:
            logs.record_log(f"Unknown order type {order_event.order_type} for {order_event.symbol}", level=3)
            return None

    def _simulate_slippage(self, price: float, order: OrderEvent, slippage_pct: Optional[float] = None) -> float:
        """
        Adjust the execution price by slippage.
        Positive slippage for BUY, negative for SELL.
        """
        slippage = slippage_pct if slippage_pct is not None else self.slippage_pct
        direction = self.DIRECTION_MAPPING.get(order.direction, 0)
        return price * (1 + direction * slippage)
