#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : test_data_handler.py.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/21 12:25

import pandas as pd
from trader.data_handler import DailyBarDataHandler
from queue import Queue
from trader.events import MarketEvent


def test_data_handler_streams_market_events(tmp_path, default_settings):
    # Create mock CSV file
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-01-01"]),
        "symbol": ["AAPL", "MSFT"],
        "open": [100, 200],
        "high": [110, 210],
        "low": [90, 190],
        "close": [105, 205]
    })
    # file = tmp_path / "mock_data.csv"
    # df.to_csv(file, index=False)

    handler = DailyBarDataHandler(data=df, settings=default_settings)
    count = 0
    while handler.continue_backtest:
        # 1. Pump next market bar

        # Pump MarketEvents from generator
        for market_event in handler.stream_next():
            if market_event is None:
                continue

            count += 1
    # assert not events.empty()
    #
    #
    # while not events.empty():
    #     event = events.get()
    #     assert isinstance(event, MarketEvent)
    #     count += 1
    #
    assert count == 4  # 2 symbols on one date
