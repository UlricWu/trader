#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : test_portfolio.py.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/21 12:23
import pytest

from trader.events import FillEvent, SignalEvent, OrderEvent
from trader.portfolio import Portfolio
from queue import Queue
from datetime import datetime


def test_portfolio_on_fill_and_equity(event_queue):
    portfolio = Portfolio(events=event_queue, initial_cash=10000)
    portfolio.update_price("AAPL", 100.0)

    signal = SignalEvent(symbol="AAPL", datetime=datetime.now(), signal_type="BUY")
    portfolio.on_signal(signal)
    assert not event_queue.empty()
    order = event_queue.get()
    assert isinstance(order, OrderEvent)

    fill = FillEvent(symbol="AAPL", price=100.0, quantity=10, direction="BUY", datetime=datetime.now())
    portfolio.on_fill(fill)
    assert portfolio.holdings["AAPL"] == 10
    assert portfolio.cash == 9000
    assert round(portfolio.equity, 2) == 10000.0


def test_portfolio_stats(event_queue):
    portfolio = Portfolio(events=event_queue, initial_cash=10000)
    portfolio.update_price("AAPL", 150.0)
    stats = portfolio.stats
    assert "cash" in stats
    assert "holdings" in stats
    assert "current_prices" in stats
    assert "equity" in stats


def test_portfolio_buy_and_sell(event_queue):
    p = Portfolio(event_queue, initial_cash=1000)

    # Set price and simulate BUY signal
    p.update_price("AAPL", 10)
    signal = SignalEvent("AAPL", datetime(2023, 1, 1), "BUY")
    p.on_signal(signal)

    order_event = event_queue.get()
    assert isinstance(order_event, OrderEvent)
    assert order_event.direction == "BUY"

    fill = FillEvent("AAPL", 10, 10, "BUY", datetime(2023, 1, 1))
    p.on_fill(fill)

    assert p.holdings["AAPL"] == 10
    assert p.cash == 1000 - 10 * 10

    # Sell
    signal = SignalEvent("AAPL", datetime(2023, 1, 2), "SELL")
    p.on_signal(signal)
    fill_sell = FillEvent("AAPL", 10, 10, "SELL", datetime(2023, 1, 2))
    p.on_fill(fill_sell)

    assert p.holdings["AAPL"] == 0
    assert p.cash == 1000


def test_portfolio_sell_without_position(event_queue):
    portfolio = Portfolio(events=event_queue, initial_cash=10000)
    portfolio.update_price("GOOG", 200.0)
    signal = SignalEvent(symbol="GOOG", datetime=datetime.now(), signal_type="SELL")
    portfolio.on_signal(signal)
    assert event_queue.empty()  # No order created if holdings < quantity


def test_portfolio_buy_without_enough_cash(event_queue):
    portfolio = Portfolio(events=event_queue, initial_cash=100)
    portfolio.update_price("AMZN", 1000.0)
    signal = SignalEvent(symbol="AMZN", datetime=datetime.now(), signal_type="BUY")
    portfolio.on_signal(signal)
    assert event_queue.empty()  # No order due to insufficient cash


def test_buy_commission(Commission_portfolio_with_mock_events):
    portfolio, events = Commission_portfolio_with_mock_events
    buy_order = FillEvent(symbol='AAPL', price=100, quantity=10, direction="BUY", datetime='2025-04-22')

    # Apply buy fill
    portfolio.on_fill(buy_order)

    expected_commission = portfolio.calculate_buy_commission(100 * 10)

    assert portfolio.cash == 100000 - (100 * 10 + expected_commission)
    assert portfolio.holdings['AAPL'] == 10  # Check that the holding quantity increased


def test_sell_commission(Commission_portfolio_with_mock_events):
    portfolio, events = Commission_portfolio_with_mock_events
    sell_order = FillEvent(symbol='AAPL', price=100, quantity=10, direction="SELL", datetime='2025-04-22')

    # Add initial holdings to be able to sell
    portfolio.holdings['AAPL'] = 10

    # Apply sell fill
    portfolio.on_fill(sell_order)

    expected_commission = portfolio.calculate_sell_commission(100 * 10)

    assert portfolio.cash == 100000 + (100 * 10 - expected_commission)
    assert portfolio.holdings['AAPL'] == 0  # Check that the holding quantity decreased


def test_sell_with_stamp_duty_and_transfer_fee(Commission_portfolio_with_mock_events):
    portfolio, events = Commission_portfolio_with_mock_events
    portfolio.holdings['AAPL'] = 10  # Ensure there are holdings to sell

    # This is to test the stamp duty and transfer fee
    sell_order = FillEvent(symbol='AAPL', price=100, quantity=10, direction="SELL", datetime='2025-04-22')

    # Apply sell fill
    portfolio.on_fill(sell_order)

    # Calculate expected commission, stamp duty, and transfer fee
    expected_commission = portfolio.calculate_sell_commission(100 * 10)

    # Check if the sell correctly deducts the commission + fees
    assert portfolio.cash == 100000 + (100 * 10 - expected_commission)
    assert portfolio.holdings['AAPL'] == 0  # Ensure holding quantity is now 0


def test_on_fill_buy_commission_applied(Commission_portfolio_with_mock_events):
    portfolio, events = Commission_portfolio_with_mock_events
    fill = FillEvent(symbol="AAPL", price=100, quantity=10, direction="BUY", datetime="2025-04-22")
    portfolio.on_fill(fill)
    expected_commission = max(1000 * 0.0003, 5.0)
    assert pytest.approx(portfolio.cash, 0.01) == 100000 - 1000 - expected_commission
    assert portfolio.holdings["AAPL"] == 10

def test_on_fill_sell_commission_stamp_transfer_applied(Commission_portfolio_with_mock_events):
    portfolio, events = Commission_portfolio_with_mock_events
    portfolio.holdings["AAPL"] = 10
    fill = FillEvent(symbol="AAPL", price=100, quantity=10, direction="SELL", datetime="2025-04-22")
    portfolio.on_fill(fill)

    gross_proceeds = 1000
    commission = max(gross_proceeds * 0.0003, 5.0)
    stamp = gross_proceeds * 0.001
    transfer = gross_proceeds * 0.00001
    net_proceeds = gross_proceeds - commission - stamp - transfer

    assert pytest.approx(portfolio.cash, 0.01) == 100000 + net_proceeds
    assert portfolio.holdings["AAPL"] == 0
