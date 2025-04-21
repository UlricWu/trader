# execution.py
import random

import pandas as pd

from trader.events import FillEvent, EventType
from utilts.logs import logs


class ExecutionHandler:
    mapping = {
        "BUY": 1,
        "SELL": -1
    }

    def __init__(self, events, slippage=0.0):
        self.events = events
        self.slippage_pct = slippage

    def execute_order(self, order, price):
        if order.type != EventType.ORDER:
            message = f"Order type={order.type} != EventType.ORDER={EventType.ORDER} not implemented"
            logs.record_log(message=message, level=3)
            return

        if order.order_type == "MKT":
            executed_price = self.simulate_slippage(price, order)

        elif order.order_type == "LIMIT":
            if order.direction == "BUY" and price > order.limit_price:
                message = f"Order for {order.symbol} did not meet the limit price, waiting for better conditions."
                logs.record_log(message=message, level=3)
                return  # Can't execute at worse price
            if order.direction == "SELL" and price < order.limit_price:
                message = f"Order for {order.symbol} did not meet the limit price, waiting for better conditions."
                logs.record_log(message=message, level=3)
                return
            executed_price = order.limit_price
        else:
            message = f"Order for {order.symbol} not in order.order_type"
            logs.record_log(message=message, level=3)
            return

        fill = FillEvent(
            symbol=order.symbol,
            price=executed_price,
            quantity=order.quantity,
            direction=order.direction,
            datetime=order.datetime
        )
        logs.record_log(f"Order for {repr(order)} with executed_price={executed_price} ")
        self.events.put(fill)

    def simulate_slippage(self, price, order, slippage_pct=None):
        """Slippage occurs when there is a discrepancy between the expected price and the actual execution price."""

        # slippage = random.uniform(-slippage_pct)
        slippage_pct = slippage_pct if slippage_pct else self.slippage_pct
        direction = self.mapping.get(order.direction, 0)
        return price * (1 + direction * slippage_pct)
