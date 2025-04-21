#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : conftest.py.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/21 13:14


# tests/conftest.py
import pytest
import queue
from trader.events import EventType, MarketEvent
from trader.data_handler import DailyBarDataHandler
from trader.strategy import Strategy
from trader.execution import SimulatedExecutionHandler
from trader.portfolio import Portfolio
from datetime import datetime
from unittest.mock import MagicMock


@pytest.fixture
def event_queue():
    return queue.Queue()


@pytest.fixture
def mock_strategy(event_queue):
    return Strategy(events=event_queue)


@pytest.fixture
def mock_execution_handler(event_queue):
    return SimulatedExecutionHandler(events=event_queue)


@pytest.fixture
def mock_portfolio(event_queue):
    return Portfolio(events=event_queue)


# @pytest.fixture
# def mock_data(tmp_path, event_queue):
#     data = """Date,Symbol,Open,High,Low,Close
# 2023-01-01,AAPL,100,105,95,102
# 2023-01-02,AAPL,102,108,101,107
# 2023-01-03,AAPL,107,109,106,108
# 2023-01-01,MSFT,200,205,195,202
# 2023-01-02,MSFT,202,208,201,207
# 2023-01-03,MSFT,207,209,206,208
# """
#     file_path = tmp_path / "mock_data.csv"
#     file_path.write_text(data)
#     return DailyBarDataHandler(str(file_path), event_queue)

@pytest.fixture
def sample_market_event():
    return MarketEvent(
        datetime=datetime(2023, 1, 1),
        symbol="AAPL",
        open=100.0,
        high=105.0,
        low=99.0,
        close=102.0
    )


@pytest.fixture
def mock_event_queue():
    """Unit testing individual components
        Asserting method calls or parameters
        Testing multiple interactions
    """
    return MagicMock()
