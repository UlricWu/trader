#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : test_backtest.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/21 10:55
import pandas as pd
import pytest

from trader.backtest_engine import Backtest


def test_backtest_initializes_components(default_settings):
    # patch_components(bars=1)
    bt = Backtest(data=[], settings=default_settings)  # data is unused by fakes
    assert hasattr(bt, "events")
    assert hasattr(bt, "data_handler")
    assert hasattr(bt, "strategy")
    assert hasattr(bt, "execution_handler")
    assert hasattr(bt, "portfolio")


def test_backtest_equity(setup_backtest, default_settings):
    # Run the backtest
    backtest = Backtest(data=setup_backtest, settings=default_settings)
    # backtest = setup_backtest
    backtest.run()

    # Test that equity has changed (it should start with some initial value)
    assert len(backtest.portfolio.history) > 0, "Equity history should not be empty"

    # Check that equity starts with initial cash, and increases due to buying/selling
    initial_equity = backtest.portfolio.history[0][1]
    final_equity = backtest.portfolio.history[-1][1]
    # print(backtest.portfolio.history)
    #
    assert final_equity > initial_equity, "Final equity should be greater than initial equity"
