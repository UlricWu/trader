#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : test_execution.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/21 13:18
from trader.execution import ExecutionHandler
from trader.events import OrderEvent, FillEvent, EventType
from datetime import datetime
from trader.execution import ExecutionHandler


def test_execution_handler(event_queue, event_execution_handler, default_settings):
    order = OrderEvent(symbol="AAPL", order_type="MKT", quantity=5, direction="BUY", datetime=datetime.now())
    event_execution_handler.execute_order(order, market_price=100.0)
    assert not event_queue.empty()
    fill = event_queue.get()
    assert isinstance(fill, FillEvent)
    assert fill.symbol == "AAPL"
    assert fill.price == 100.0 * (1 + default_settings.trading.SLIPPAGE)


def test_market_order_slippage_applied(mock_event_queue, default_settings):
    execution_handler = ExecutionHandler(mock_event_queue, settings=default_settings)
    order = OrderEvent(
        symbol="AAPL",
        order_type="MKT",
        quantity=10,
        direction="BUY",
        datetime=datetime(2023, 1, 1)
    )
    price = 150.0
    execution_handler.execute_order(order, price)
    # Verify it was called
    assert mock_event_queue.put.call_count == 1

    fill_event = mock_event_queue.put.call_args[0][0]
    assert isinstance(fill_event, FillEvent)
    assert fill_event.price == price * (1 + default_settings.trading.SLIPPAGE)  # slippage applied for MKT


def test_execute_order_with_slippage_sell(mock_event_queue, default_settings):
    slippage = 0.05
    execution_handler = ExecutionHandler(mock_event_queue, settings=default_settings)
    order = OrderEvent(
        symbol="AAPL",
        order_type="MKT",
        quantity=5,
        direction="SELL",
        datetime=datetime(2024, 1, 1)
    )
    order.type = EventType.ORDER

    execution_handler.execute_order(order, market_price=150.0)

    fill_event = mock_event_queue.put.call_args[0][0]
    assert isinstance(fill_event, FillEvent)
    assert fill_event.price == 150 * (1 - default_settings.trading.SLIPPAGE)  # SELL - slippage
    assert fill_event.direction == "SELL"


def test_execution_handler_generates_fill_with_real_queue(event_queue, event_execution_handler, default_settings):
    order = OrderEvent(
        symbol="AAPL",
        order_type="MKT",
        quantity=10,
        direction="BUY",
        datetime=datetime(2025, 1, 1)
    )

    event_execution_handler.execute_order(order, market_price=100.0)

    # Assert the FillEvent is in the queue
    assert not event_queue.empty()
    event = event_queue.get()
    assert isinstance(event, FillEvent)
    assert event.symbol == "AAPL"
    assert event.direction == "BUY"
    assert event.price == 100.0 * (1 + default_settings.trading.SLIPPAGE)


def test_limit_order_sell_no_slippage(mock_event_queue, default_settings):
    price = 100.0
    mock_execution_handler = ExecutionHandler(mock_event_queue, settings=default_settings)
    order = OrderEvent("AAPL", "LIMIT", 10, "SELL", datetime(2024, 1, 1), price)
    order.type = EventType.ORDER

    mock_execution_handler.execute_order(order, market_price=price)

    fill = mock_event_queue.put.call_args[0][0]
    assert fill.price == price
    assert fill.direction == "SELL"


# def test_limit_order_fills_when_price_matches(event_execution_handler, mock_event_queue):
#     # Example daily bar
#     daily_bar = MarketEvent(
#         datetime=datetime(2024, 1, 1),
#         symbol="TEST",
#         open=100,
#         high=105,
#         low=95,
#         close=102
#     )
#
#     # Create a BUY LIMIT order at price 96, should be filled since low=95
#     limit_order = OrderEvent(
#         symbol="TEST",
#         order_type="LIMIT",
#         quantity=10,
#         direction="BUY",
#         limit_price=96,
#         datetime=daily_bar.datetime
#     )
#
#     # Simulate limit order execution
#     event_execution_handler.execute_order(order=limit_order, daily_bar)
#
#     # Assert a fill was put into the queue
#     assert mock_event_queue.put.called
#     fill_event = mock_event_queue.put.call_args[0][0]
#     assert fill_event.type == EventType.FILL
#     assert fill_event.symbol == "TEST"
#     assert fill_event.fill_price == 96
#     assert fill_event.quantity == 10
#     assert fill_event.direction == "BUY"

