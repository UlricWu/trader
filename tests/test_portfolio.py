#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : test_portfolio.py.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/21 12:23

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
    assert p.cash == 1000 - 10*10

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

