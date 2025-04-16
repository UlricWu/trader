#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : test_strategy.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/16 17:31

import pandas as pd
import numpy as np
import pytest
from trader.backtest import Strategy


@pytest.fixture
def sample_data():
    data = []

    days = 150

    date = pd.date_range(start="2020-01-01", periods=days)
    for i in range(days):
        data.append({"date": date[i],
                     "symbol": "AAPL",
                     "close": 100 + i * 0.5})
    df = pd.DataFrame(data)
    return df


def test_strategy_signal_generation(sample_data):
    # config = StrategyConfig(window_size=20, long_threshold=0.005, short_threshold=-0.005)
    strategy = Strategy()
    df = strategy.generate_signals(sample_data)

    # Check column exists
    assert "signal" in df.columns

    # Check that at least some signals are not 0
    # assert df["signal"].abs().sum() > 0  # todo

    # Check first N signals are zero (before training starts)
    window_size = 100
    initial_signals = df["signal"].iloc[:window_size]
    assert all(initial_signals == 0)
