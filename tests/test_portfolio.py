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

def test_portfolio_buy_and_sell():
    events = Queue()
    p = Portfolio(events, initial_cash=1000)

    # Set price and simulate BUY signal
    p.update_price("AAPL", 10)
    signal = SignalEvent("AAPL", datetime(2023, 1, 1), "BUY")
    p.on_signal(signal)

    order_event = events.get()
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

