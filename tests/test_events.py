#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : test_events.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/21 13:16

from trader.events import MarketEvent, EventType, OrderEvent, SignalEvent
from datetime import datetime
from unittest.mock import MagicMock
from trader.strategy import Strategy
from trader.execution import ExecutionHandler


def test_market_event(sample_market_event):
    assert sample_market_event.symbol == "AAPL"
    assert sample_market_event.close == 102.0


def test_simple_strategy_emits_signal_after_window(mock_event_queue):
    strategy = Strategy(mock_event_queue, window=3)

    # Create 3 market events with increasing prices
    events = [
        MarketEvent(datetime(2024, 1, 1), "AAPL", 100, 101, 99, 100),
        MarketEvent(datetime(2024, 1, 2), "AAPL", 101, 102, 100, 101),
        MarketEvent(datetime(2024, 1, 3), "AAPL", 102, 103, 101, 105),  # > average(100,101,102) => BUY
    ]

    for event in events:
        strategy.on_market(event)

    # Ensure a SignalEvent was placed into the queue
    assert mock_event_queue.put.called
    args, kwargs = mock_event_queue.put.call_args
    signal = args[0]
    assert signal.type == EventType.SIGNAL
    assert signal.symbol == "AAPL"
    assert signal.signal_type == "BUY"


def test_execution_handler_puts_fill_event(mock_event_queue):
    handler = ExecutionHandler(mock_event_queue)

    order = OrderEvent("AAPL", "MKT", 10, "BUY", datetime(2024, 1, 1))
    handler.execute_order(order, market_price=150.0)

    assert mock_event_queue.put.called
    args, _ = mock_event_queue.put.call_args
    fill = args[0]
    assert fill.type == EventType.FILL
    assert fill.symbol == "AAPL"
    assert fill.price == 150.0
    assert fill.quantity == 10
    assert fill.direction == "BUY"


def test_strategy_generates_signal(mock_event_queue):
    strategy = Strategy(mock_event_queue, window=2)

    # Feed two events to trigger a signal
    event1 = MarketEvent(datetime(2024, 1, 1), "AAPL", 100, 101, 99, 100)
    event2 = MarketEvent(datetime(2024, 1, 2), "AAPL", 102, 103, 101, 102)

    strategy.on_market(event1)
    strategy.on_market(event2)

    assert mock_event_queue.put.called
