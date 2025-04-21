#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : test_backtest.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/21 10:55
import pandas as pd
import pytest

from trader.backtest_engine import Backtest
from datetime import datetime, timedelta

@pytest.fixture
def setup_backtest():
    # Generate the mock data for testing
    data = []
    base_date = datetime(2023, 1, 1)
    for i in range(5):  # 5 days of data
        date = base_date + timedelta(days=i)
        open_price = 100 + i
        close_price = open_price + 1
        high_price = close_price + 1
        low_price = open_price - 1
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "symbol": "AAPL",
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price
        })

    mock_data = pd.DataFrame(data)

    mock_data["date"] = pd.to_datetime(mock_data["date"])
    return mock_data

    # # Initialize the backtest class
    # backtest = Backtest(data=mock_data)
    # return backtest




def test_backtest_equity(setup_backtest):
    # Run the backtest
    backtest = Backtest(data=setup_backtest)
    # backtest = setup_backtest
    backtest.run()

    # Test that equity has changed (it should start with some initial value)
    assert len(backtest.portfolio.history) > 0, "Equity history should not be empty"

    # Check that equity starts with initial cash, and increases due to buying/selling
    initial_equity = backtest.portfolio.history[0][1]
    final_equity = backtest.portfolio.history[-1][1]
    #
    assert final_equity > initial_equity, "Final equity should be greater than initial equity"
