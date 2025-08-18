#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : test_portfolio.py.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/21 12:23
import pandas as pd
import pytest

from trader.events import FillEvent, SignalEvent, OrderEvent
from trader.portfolio import Portfolio, Position
from queue import Queue
from datetime import datetime


def test_buy_with_commission(Commission_portfolio_with_mock_events):
    portfolio, _ = Commission_portfolio_with_mock_events
    fill = FillEvent(symbol="AAPL", price=100, quantity=10, direction="BUY", datetime="2025-04-22")
    # fill = FillEvent("AAPL", datetime(2025, 1, 1), price=100.0, qty=10, direction="BUY")
    portfolio.on_fill(fill)

    # Expected cost = 100 * 10 = 1000
    # Commission = max(1000 * 0.001, 5) = 5
    expected_cash = 100000.0 - 1000 - 5

    assert abs(portfolio.cash - expected_cash) < 1e-6
    assert portfolio.positions["AAPL"].quantity == 10
    assert "AAPL" not in portfolio.realized_pnl  # No realized PnL on buy


def test_china_buy_commission_min(event_portfolio):
    fill = FillEvent(symbol="XYZ", price=1.0, quantity=1, direction="BUY", datetime="2025-04-22")
    event_portfolio.on_fill(fill)

    expected_cash = 100000.0 - 1.0 - 5.0  # min commission ¥5
    assert abs(event_portfolio.cash - expected_cash) < 1e-6
    assert event_portfolio.positions["XYZ"].quantity == 1


def test_china_sell_commission_and_pnl(event_portfolio):
    """
    Buy 100 @ ¥10, Sell 100 @ ¥12, China fees:
    Buy Commission = ¥10, Sell Commission = ¥12, Stamp = ¥1.2, Transfer = ¥0.012
    Realized PnL = 200 - 10 - (12+1.2+0.012) = 176.788
    """

    # Buy
    buy_fill = FillEvent(symbol="AAPL", price=10, quantity=100, direction="BUY", datetime="2025-04-22")

    event_portfolio.on_fill(buy_fill)

    # Sell
    sell_fill = FillEvent(symbol="AAPL", price=12, quantity=100, direction="SELL", datetime="2025-04-23")

    event_portfolio.on_fill(sell_fill)

    # Calculate fees
    buy_commission = max(1000 * 0.001, 5.0)  # 5
    sell_commission = max(1200 * 0.001, 5.0)  # 5
    stamp_duty = 1200 * 0.001  # 1.2
    transfer_fee = 1200 * 0.00001  # 0.012
    total_sell_fee = sell_commission + stamp_duty + transfer_fee  # ≈5+1.2+0.012=6.212

    gross_pnl = (12.0 - 10.0) * 100  # 200
    net_pnl = gross_pnl - buy_commission - total_sell_fee  # ≈200-5-6.212=188.788

    realized_pnl = event_portfolio.realized_pnl["AAPL"]

    assert abs(realized_pnl - net_pnl) < 1e-3
    assert event_portfolio.positions["AAPL"].quantity == 0
    assert abs(event_portfolio.cash - (100000.0 - 1000 - buy_commission + 1200 - total_sell_fee)) < 1e-3


def test_sell_commission_and_all_fees(event_portfolio):
    # Buy 100 @ 10, Sell 100 @ 12
    buy_fill = FillEvent(symbol="AAPL", price=10.0, quantity=100, direction="BUY", datetime="2025-04-22")
    event_portfolio.on_fill(buy_fill)

    sell_fill = FillEvent(symbol="AAPL", price=12.0, quantity=100, direction="SELL", datetime="2025-04-23")
    event_portfolio.on_fill(sell_fill)

    # Fees
    buy_commission = max(1000 * 0.001, 5)  # ¥10
    sell_commission = max(1200 * 0.001, 5)  # ¥12
    stamp_duty = 1200 * 0.001  # ¥1.2
    transfer_fee = 1200 * 0.00001  # ¥0.012
    total_sell_fee = sell_commission + stamp_duty + transfer_fee  # ≈13.212

    gross_pnl = 200
    net_pnl = gross_pnl - buy_commission - total_sell_fee  # ≈176.788

    realized = event_portfolio.realized_pnl["AAPL"]
    assert abs(realized - net_pnl) < 1e-3

    expected_cash = (
            100000.0 - 1000.0 - buy_commission + 1200.0 - total_sell_fee
    )
    assert abs(event_portfolio.cash - expected_cash) < 1e-3
    assert event_portfolio.positions["AAPL"].quantity == 0


def test_sell_with_commission_and_pnl(Commission_portfolio_with_mock_events):
    portfolio, _ = Commission_portfolio_with_mock_events
    # Buy first
    buy_fill = FillEvent(symbol="AAPL", price=100, quantity=10, direction="BUY", datetime="2025-04-22")
    portfolio.on_fill(buy_fill)

    # Sell later at higher price
    sell_fill = FillEvent(symbol="AAPL", price=110, quantity=10, direction="SELL", datetime="2025-04-23")
    portfolio.on_fill(sell_fill)

    # Buy Commission = 5
    # Sell Commission = max(1100 * 0.001, 5) = 5 + stamp + transfer = ~6.112
    sell_fee = max(1100 * 0.001, 5) + 1100 * 0.001 + 1100 * 0.00001  # ≈ 6.112

    gross_pnl = (110 - 100) * 10  # 100
    net_pnl = gross_pnl - 5 - sell_fee

    realized_pnl = portfolio.realized_pnl["AAPL"]
    assert abs(realized_pnl - net_pnl) < 1e-3
    assert portfolio.positions["AAPL"].quantity == 0  # position closed


def test_realized_pnl_partial_sell_with_proportional_fee(event_portfolio):
    # Buy 100 @ 10, Sell 50 @ 12
    event_portfolio.on_fill(FillEvent(symbol="TSLA", price=10.0, quantity=100, direction="BUY", datetime="2025-04-22"))
    event_portfolio.on_fill(FillEvent(symbol="TSLA", price=12.0, quantity=50, direction="SELL", datetime="2025-04-23"))

    buy_commission = max(1000 * 0.001, 5)  # ¥10
    sell_commission = max(600 * 0.001, 5)  # ¥6
    stamp_duty = 600 * 0.001  # ¥0.6
    transfer_fee = 600 * 0.00001  # ¥0.006
    total_sell_fee = sell_commission + stamp_duty + transfer_fee  # ~6.606

    proportional_buy_fee = (50 / 100) * buy_commission  # ¥5
    gross_pnl = 100
    net_pnl = gross_pnl - proportional_buy_fee - total_sell_fee  # ~88.394

    realized = event_portfolio.realized_pnl["TSLA"]
    assert abs(realized - net_pnl) < 1e-3
    assert event_portfolio.positions["TSLA"].quantity == 50


def test_multiple_symbols_independent_pnl(event_portfolio):
    # Buy/Sell AAPL
    event_portfolio.on_fill(FillEvent(symbol="AAPL", price=10.0, quantity=100, direction="BUY", datetime="2025-04-22"))
    event_portfolio.on_fill(FillEvent(symbol="AAPL", price=11.0, quantity=100, direction="SELL", datetime="2025-04-23"))

    # Buy/Sell MSFT
    event_portfolio.on_fill(FillEvent(symbol="MSFT", price=20.0, quantity=50, direction="BUY", datetime="2025-04-24"))
    event_portfolio.on_fill(FillEvent(symbol="MSFT", price=22.0, quantity=50, direction="SELL", datetime="2025-04-25"))

    # AAPL PnL
    aapl_buy_fee = max(1000 * 0.001, 5)  # ¥10
    aapl_sell_fee = max(1100 * 0.001, 5) + 1.1 + 0.011  # ~12.111
    aapl_net_pnl = 100 - aapl_buy_fee - aapl_sell_fee

    # MSFT PnL
    msft_buy_fee = max(1000 * 0.001, 5)  # ¥10
    msft_sell_fee = max(1100 * 0.001, 5) + 1.1 + 0.011  # ~12.111
    msft_net_pnl = 100 - msft_buy_fee - msft_sell_fee

    assert abs(event_portfolio.realized_pnl["AAPL"] - aapl_net_pnl) < 1e-3
    assert abs(event_portfolio.realized_pnl["MSFT"] - msft_net_pnl) < 1e-3
#
# def test_portfolio_on_fill_and_equity(default_settings):
#     portfolio = Portfolio(settings=default_settings)
#     portfolio.update_price("AAPL", 100.0)
#
#     signal = SignalEvent(symbol="AAPL", datetime=datetime.now(), signal_type="BUY")
#
#     # assert not event_queue.empty()
#     order = portfolio.on_signal(signal)
#     assert isinstance(order, OrderEvent)
#
#     fill = FillEvent(symbol="AAPL", price=100.0, quantity=10, direction="BUY", datetime=datetime.now())
#     portfolio.on_fill(fill)
#     # assert portfolio.holdings["AAPL"] == 10
#     assert portfolio.cash == 9000
#     assert round(portfolio.equity, 2) == 10000.0
# #
#
# # def test_portfolio_stats(event_queue):
# #     portfolio = Portfolio(events=event_queue, initial_cash=10000)
# #     portfolio.update_price("AAPL", 150.0)
# #     stats = portfolio.stats
# #     assert "cash" in stats
# #     assert "holdings" in stats
# #     assert "current_prices" in stats
# #     assert "equity" in stats
#
#
# # def test_portfolio_buy_and_sell(event_queue):
# #     p = Portfolio(event_queue, initial_cash=1000)
# #
# #     # Set price and simulate BUY signal
# #     p.update_price("AAPL", 10)
# #     signal = SignalEvent("AAPL", datetime(2023, 1, 1), "BUY")
# #     p.on_signal(signal)
# #
# #     order_event = event_queue.get()
# #     assert isinstance(order_event, OrderEvent)
# #     assert order_event.direction == "BUY"
# #
# #     fill = FillEvent("AAPL", 10, 10, "BUY", datetime(2023, 1, 1))
# #     p.on_fill(fill)
# #
# #     assert p.holdings["AAPL"] == 10
# #     assert p.cash == 1000 - 10 * 10
# #
# #     # Sell
# #     signal = SignalEvent("AAPL", datetime(2023, 1, 2), "SELL")
# #     p.on_signal(signal)
# #     fill_sell = FillEvent("AAPL", 10, 10, "SELL", datetime(2023, 1, 2))
# #     p.on_fill(fill_sell)
# #
# #     assert p.holdings["AAPL"] == 0
# #     assert p.cash == 1000
#
#
# # def test_portfolio_sell_without_position(event_queue):
# #     portfolio = Portfolio(events=event_queue, initial_cash=10000)
# #     portfolio.update_price("GOOG", 200.0)
# #     signal = SignalEvent(symbol="GOOG", datetime=datetime.now(), signal_type="SELL")
# #     portfolio.on_signal(signal)
# #     assert event_queue.empty()  # No order created if holdings < quantity
#
#
# # def test_portfolio_buy_without_enough_cash(event_queue):
# #     portfolio = Portfolio(events=event_queue, initial_cash=100)
# #     portfolio.update_price("AMZN", 1000.0)
# #     signal = SignalEvent(symbol="AMZN", datetime=datetime.now(), signal_type="BUY")
# #     portfolio.on_signal(signal)
# #     assert event_queue.empty()  # No order due to insufficient cash
#
#
def test_buy_commission(Commission_portfolio_with_mock_events):
    portfolio, _ = Commission_portfolio_with_mock_events
    buy_order = FillEvent(symbol='AAPL', price=100, quantity=10, direction="BUY", datetime='2025-04-22')

    # Apply buy fill
    portfolio.on_fill(buy_order)

    expected_commission = portfolio.calculate_buy_commission(100 * 10)

    assert portfolio.cash == 100000 - (100 * 10 + expected_commission)
    assert portfolio.positions['AAPL'].quantity == 10  # Check that the holding quantity increased

