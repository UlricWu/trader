#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : test_execution.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/21 13:18
from trader.execution import ExecutionHandler
from trader.events import OrderEvent, FillEvent
from datetime import datetime


def test_execution_handler(event_queue, mock_execution_handler):
    order = OrderEvent(symbol="AAPL", order_type="MKT", quantity=5, direction="BUY", datetime=datetime.now())
    mock_execution_handler.execute_order(order, price=100.0)
    assert not event_queue.empty()
    fill = event_queue.get()
    assert isinstance(fill, FillEvent)
    assert fill.symbol == "AAPL"
    assert fill.price == 100.0


def test_execution_handler_generates_fill_with_real_queue(event_queue, mock_execution_handler):
    order = OrderEvent(
        symbol="AAPL",
        order_type="MKT",
        quantity=10,
        direction="BUY",
        datetime=datetime(2025, 1, 1)
    )

    mock_execution_handler.execute_order(order, price=100.0)

    # Assert the FillEvent is in the queue
    assert not event_queue.empty()
    event = event_queue.get()
    assert isinstance(event, FillEvent)
    assert event.symbol == "AAPL"
    assert event.direction == "BUY"
    assert event.price == 100.0

