# execution.py
from events import OrderEvent, FillEvent
class SimulatedExecutionHandler:
    def __init__(self, events):
        self.events = events

    def on_signal(self, signal_event):
        order = OrderEvent(
            datetime=signal_event.datetime,
            symbol=signal_event.symbol,
            quantity=100,
            direction=signal_event.signal_type
        )
        self.events.put(order)

    def on_order(self, order_event, price):
        fill = FillEvent(
            datetime=order_event.datetime,
            symbol=order_event.symbol,
            quantity=order_event.quantity,
            price=price,
            direction=order_event.direction
        )
        self.events.put(fill)
