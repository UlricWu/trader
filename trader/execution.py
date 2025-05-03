# execution.py
from __future__ import annotations

from trader.events import FillEvent, EventType, OrderEvent
from utilts.logs import logs

from trader.config import Settings


class ExecutionHandler:
    mapping = {
        "BUY": 1,
        "SELL": -1
    }

    def __init__(self, events, settings: Settings):
        self.events = events
        self.slippage_pct = settings.trading.SLIPPAGE
        self.settings = settings

    def execute_order(self, order_event: OrderEvent, market_price: float, prev_close: float | None = None) -> None:

        if order_event.type != EventType.ORDER:
            message = f"Order type={order_event.type} != EventType.ORDER={EventType.ORDER} not implemented"
            logs.record_log(message=message, level=3)
            return

        symbol = order_event.symbol
        event_direction = order_event.direction
        event_quantity = order_event.quantity

        order_type = order_event.order_type
        if order_type == "MKT":
            executed_price = self.simulate_slippage(market_price, order_event)

        # if order_event.order_type == "LIMIT":
        # if order_event.direction == "BUY" and market_event.low <= order_event.limit_price:
        #     fill_price = min(market_event.close, order_event.limit_price)  # Limit order filled at the limit price
        #     fill_event = FillEvent(
        #         symbol=symbol,
        #         price=fill_price,
        #         quantity=order_event.quantity,
        #         direction=order_event.direction,
        #         datetime=market_event.datetime
        #     )
        #     self.events.put(fill_event)
        #
        # elif order_event.direction == "SELL" and market_event.high >= order_event.limit_price:
        #     fill_price = max(market_event.close, order_event.limit_price)  # Limit order filled at the limit price
        #     fill_event = FillEvent(
        #         symbol=symbol,
        #         price=fill_price,
        #         quantity=order_event.quantity,
        #         direction=order_event.direction,
        #         datetime=market_event.datetime
        #     )

        # Execute the order unless it's blocked due to limit up/down.
        # price_data: dict with keys: open, high, low, close
        elif order_type == "LIMIT":
            if event_direction == "BUY" and market_price > order_event.limit_price:
                # logs.record_log(f"[LIMIT UP] Blocked BUY: {symbol} at {close}", 2)
                return  # Can't execute at worse price

            if event_direction == "SELL" and market_price < order_event.limit_price:
                message = f"Order for {symbol} did not meet the limit price, waiting for better conditions."
                logs.record_log(message=message, level=3)
                return

            executed_price = order_event.limit_price
        else:
            message = f"Order for {symbol} not in order.order_type"
            logs.record_log(message=message, level=3)
            return

        fill = FillEvent(
            symbol=symbol,
            price=executed_price,
            quantity=event_quantity,
            direction=event_direction,
            datetime=order_event.datetime
        )
        logs.record_log(f"Order for {repr(order_event)} with executed_price={executed_price} ")
        self.events.put(fill)

    def simulate_slippage(self, price, order, slippage_pct=None):
        """Slippage occurs when there is a discrepancy between the expected price and the actual execution price."""

        # slippage = random.uniform(-slippage_pct)
        slippage_pct = slippage_pct if slippage_pct else self.slippage_pct
        direction = self.mapping.get(order.direction, 0)
        return price * (1 + direction * slippage_pct)
