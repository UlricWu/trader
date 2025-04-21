# execution.py
import random

import pandas as pd

from trader.events import FillEvent, EventType
from utilts.logs import logs


class ExecutionHandler:
    def __init__(self, events):
        self.events = events

    def execute_order(self, order, price):
        if order.type != EventType.ORDER:
            message = f"Order type={order.type} != EventType.ORDER={EventType.ORDER} not implemented"
            logs.record_log(message=message, level=3)
            return
        fill = FillEvent(
            symbol=order.symbol,
            price=price,
            quantity=order.quantity,
            direction=order.direction,
            datetime=order.datetime
        )
        self.events.put(fill)


class SimulatedExecutionHandler(ExecutionHandler):

    def execute_order(self, order, price):
        if order.order_type == "LIMIT":
            # Only execute the order if the price meets the limit condition
            if (order.direction == "BUY" and price <= order.limit_price) or \
                    (order.direction == "SELL" and price >= order.limit_price):
                super().execute_order(order, price)
                logs.record_log(f"Order for {order.symbol} ")
            else:
                message = f"Order for {order.symbol} did not meet the limit price, waiting for better conditions."
                logs.record_log(message=message, level=3)
        elif order.order_type == "MKT":
            # Market orders are always executed at the current price
            super().execute_order(order, price)
            logs.record_log(f"Order for {order.symbol} ")
        else:
            message = f"Order for {order.symbol} not in order.order_type"
            logs.record_log(message=message, level=3)

    def simulate_slippage(self, price, slippage_pct: float = 0.02):
        """Slippage occurs when there is a discrepancy between the expected price and the actual execution price."""
        slippage = random.uniform(-slippage_pct, slippage_pct)
        return price * (1 + slippage)
