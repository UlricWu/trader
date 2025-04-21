# execution.py
import pandas as pd

from trader.events import FillEvent, EventType


class ExecutionHandler:
    def __init__(self, events):
        self.events = events

    def execute_order(self, order, price):
        if order.type != EventType.ORDER:
            print(f"Order type={order.type} != EventType.ORDER={EventType.ORDER} not implemented")
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
            else:
                print(f"Order for {order.symbol} did not meet the limit price, waiting for better conditions.")
        elif order.order_type == "MKT":
            # Market orders are always executed at the current price
            super().execute_order(order, price)
            print(f"Order for {order.symbol} ")
