#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : event_bus.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/21 11:50

# trader/event_bus.py
from collections import defaultdict
from typing import Callable, Dict, List, Tuple, Any

from collections import defaultdict
from typing import Callable, Dict, List, Type
class EventBus:
    CONSUME = object()  # sentinel to stop propagation

    def __init__(self, emit_fn: Callable[[Any], None]):
        """
        emit_fn: a function that enqueues new events (e.g., backtest.events.put)
        """
        self._handlers: Dict[Type, List[Callable]] = defaultdict(list)
        self._emit = emit_fn
        # self._subs: Dict[Type, List[Callable]] = defaultdict(list)

    def subscribe(self, event_type: str, handler: Callable, priority: int = 100):
        """
        handler signature: handler(event, bus) -> Optional[EventBus.CONSUME]
        Return EventBus.CONSUME to stop further handlers for this event.
        """
        self._handlers[event_type].append((priority, handler))
        self._handlers[event_type].sort(key=lambda x: x[0])  # lowest runs first

    def publish(self, event: Any):
        """Deliver event to subscribers in priority order."""
        if event is None or event.is_empty():
            # raise ValueError(f"empty event={event}")
            return

        handlers = self._handlers.get(event.type, [])
        for priority, h in handlers:
            result = h(event, self)

            if result is EventBus.CONSUME:
                break

    def emit(self, event: Any):
        """Enqueue a new event for later processing."""
        self._emit(event)
