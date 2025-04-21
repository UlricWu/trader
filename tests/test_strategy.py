#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : test_strategy.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/16 17:31

import pytest
from trader.events import MarketEvent, EventType
from trader.strategy import Strategy
from queue import Queue
from datetime import datetime

def test_strategy_signal_emits_buy():
    events = Queue()
    strategy = Strategy(events, window=3)

    prices = [100, 101, 102, 110]
    for price in prices:
        event = MarketEvent(datetime(2023, 1, 1), "AAPL", price, price, price, price)
        strategy.on_market(event)

    signals = []
    while not events.empty():
        event = events.get()
        if event.type == EventType.SIGNAL:
            signals.append(event.signal_type)

    assert "BUY" in signals

#
# import pandas as pd
# import numpy as np
# import pytest
# from trader.strategy import MLStrategy
#
# from configs.config import StrategyConfig
#
#
# @pytest.fixture
# def sample_data():
#     data = []
#
#     days = 50
#
#     date = pd.date_range(start="2020-01-01", periods=days)
#     for i in range(days):
#         data.append({"date": date[i],
#                      "symbol": "AAPL",
#                      "close": 100 + i * 0.5})
#     df = pd.DataFrame(data)
#     return df
#
#
# def test_strategy_signal_generation(sample_data):
#     config = StrategyConfig(window_size=20, long_threshold=0.005, short_threshold=-0.005)
#     strategy = MLStrategy(config)
#     df = strategy.generate_signals(sample_data)
#
#     # Check column exists
#     assert "signal" in df.columns
#
#     # Check that at least some signals are not 0
#     assert df["signal"].abs().sum() > 0
#
#     # Check first N signals are zero (before training starts)
#     initial_signals = df["signal"].iloc[:config.window_size]
#     assert all(initial_signals == 0)
#
#
# def test_threshold_effect(sample_data):
#     # Using high thresholds should produce mostly 0 signals
#     config = StrategyConfig(window_size=20, long_threshold=0.1, short_threshold=-0.1)
#     strategy = MLStrategy(config=config)
#     df = strategy.generate_signals(sample_data)
#
#     # Almost all signals should be 0 due to high thresholds
#     # At least 90% of the time, the model doesn’t predict extreme moves, so the signal should be 0.
#     assert 0.1 < (df["signal"] == 0).mean() < 1  # todo 0.9
#
#
# def test_low_threshold_increases_signals(sample_data):
#     config = StrategyConfig(window_size=20, long_threshold=0.1, short_threshold=-0.1)
#     strategy = MLStrategy(config=config)
#
#     df = strategy.generate_signals(sample_data)
#
#     # With tiny thresholds, we expect more trading signals
#     assert (df["signal"].abs().sum()) > 10
