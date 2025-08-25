#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : test_strategy.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/16 17:31

import pytest
from trader.events import MarketEvent, EventType, SignalEvent
from trader.strategy import RuleStrategy
from queue import Queue
from datetime import datetime


# def test_simple_strategy_signal_generation( default_settings):
#     strategy = RuleStrategy( settings=default_settings)
#     for price in [100, 101, 105, 110]:
#         event = MarketEvent(datetime=datetime(2023, 1, 1), symbol="AAPL", open=0, high=0, low=0, close=price)
#         signal = strategy.on_market(event)
#         print(signal)
#     # assert not event_queue.empty()
#     #  event_queue.get()
#     # print(signal)
#     assert isinstance(signal, SignalEvent)
#     assert signal.symbol == "AAPL"
#     assert "BUY" in signal.signal_type




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
