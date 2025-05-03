# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @File    : test_performance.py
# # @Project : trader
# # @Author  : wsw
# # @Time    : 2025/4/17 22:22
# # !test_performance.py
#
# import pytest
# import pandas as pd
# from trader.metrics import PerformanceAnalyzer
# from trader.portfolio import Fill, Portfolio


# tests/test_portfolio_performance.py

from datetime import datetime
from trader.events import FillEvent, MarketEvent
from trader.performance import PerformanceAnalyzer

# tests/test_performance_analyzer.py

from trader.performance import PerformanceAnalyzer


def test_portfolio_performance_metrics(Commission_portfolio_with_mock_events):
    portfolio, _ = Commission_portfolio_with_mock_events

    symbol = "000001.SZ"

    # # Day 1 - price update
    portfolio.update_price(MarketEvent(datetime=datetime(2024, 1, 1), symbol=symbol, open=10, high=10, low=10, close=10)
                           )
    portfolio.history.append((datetime(2024, 1, 1), 100000))

    # Day 2 - Buy
    fill_buy = FillEvent(
        symbol=symbol,
        datetime=datetime(2024, 1, 2),
        price=10.0,
        quantity=1000,
        direction="BUY"
    )
    portfolio.on_fill(fill_buy)
    portfolio.history.append((datetime(2024, 1, 2), 90000))

    # Day 3 - Sell
    fill_sell = FillEvent(
        symbol=symbol,
        datetime=datetime(2024, 1, 3),
        price=12.0,
        quantity=1000,
        direction="SELL"
    )
    portfolio.on_fill(fill_sell)
    portfolio.history.append((datetime(2024, 1, 3), 102000))

    metrics = PerformanceAnalyzer(portfolio).summary()

    assert metrics["initial_cash"] == 100000
    assert metrics["final_equity"] == 102000
    assert round(metrics["total_return"], 5) == 0.02
    assert metrics["num_trades"] == 1
    assert metrics["win_rate"] == 1.0
    assert metrics["average_win"] == 2000.0
    assert metrics["average_loss"] == 0.0
    assert metrics["profit_factor"] >= 999999999.0
    assert metrics["per_symbol"][symbol] == {'symbol': '000001.SZ', 'realized_pnl': 2000.0, 'win': 1, 'loss': 0, 'neutral': 0}

def test_performance_analyzer_with_mixed_trades(Commission_portfolio_with_mock_events):
    portfolio, _ = Commission_portfolio_with_mock_events
    portfolio.history = [
        (datetime(2024, 1, 1), 100000.0),
        (datetime(2024, 1, 2), 96000.0),
        (datetime(2024, 1, 3), 102000.0),
    ]
    portfolio.realized_pnl = {
        "000001.SZ": 2000.0,
        "000002.SZ": -4000.0,
    }

    analyzer = PerformanceAnalyzer(portfolio)
    metrics = analyzer.summary()

    assert metrics["initial_cash"] == 100000.0
    assert metrics["final_equity"] == 102000.0
    assert metrics["total_return"] == 0.02
    assert metrics["num_trades"] == 2
    assert metrics["win_rate"] == 0.5
    assert metrics["average_win"] == 2000.0
    assert metrics["average_loss"] == 4000.0
    assert metrics["profit_factor"] == 0.5

def test_performance_with_win_loss_neutral(Commission_portfolio_with_mock_events):
    portfolio ,_ =Commission_portfolio_with_mock_events
    portfolio.history = [
        (datetime(2024, 1, 1), 100000.0),
        (datetime(2024, 1, 2), 98000.0),
        (datetime(2024, 1, 3), 102000.0),
    ]
    portfolio.realized_pnl = {
        "000001.SZ": 2000.0,
        "000002.SZ": -4000.0,
        "000003.SZ": 0.0,
    }

    analyzer = PerformanceAnalyzer(portfolio)
    summary = analyzer.summary()

    assert summary["num_trades"] == 3
    assert summary["win_rate"] == round(1 / 3, 4)
    assert summary["average_win"] == 2000.0
    assert summary["average_loss"] == 4000.0
    assert summary["profit_factor"] == 0.5

    assert summary["per_symbol"]["000001.SZ"]["win"] == 1
    assert summary["per_symbol"]["000002.SZ"]["loss"] == 1
    assert summary["per_symbol"]["000003.SZ"]["neutral"] == 1


# @pytest.fixture
# def sample_fills_portfolio():
#     portfolio = Portfolio()
#
#     # AAPL trades
#     portfolio.update_position(Fill(symbol="AAPL", date=pd.to_datetime("2024-01-01"), price=100, quantity=10))  # $1000
#     portfolio.update_position(Fill(symbol="AAPL", date=pd.to_datetime("2024-01-02"), price=110, quantity=10))  # $1100
#     portfolio.update_position(Fill(symbol="AAPL", date=pd.to_datetime("2024-01-03"), price=105, quantity=10))  # $1050
#
#     # MSFT trades
#     portfolio.update_position(Fill(symbol="MSFT", date=pd.to_datetime("2024-01-01"), price=200, quantity=5))  # $1000
#     portfolio.update_position(Fill(symbol="MSFT", date=pd.to_datetime("2024-01-02"), price=202, quantity=5))  # $1010
#     portfolio.update_position(Fill(symbol="MSFT", date=pd.to_datetime("2024-01-03"), price=198, quantity=5))  # $990
#
#     return portfolio
#
#
# def test_performance_from_fills(sample_fills_portfolio):
#     analyzer = PerformanceAnalyzer(sample_fills_portfolio)
#     summary = analyzer.summary()
#
#     symbol = summary['symbol'].unique().tolist()
#     assert isinstance(summary, pd.DataFrame)
#     assert "AAPL" in symbol
#     assert "MSFT" in symbol
#
#     # Check return is calculated correctly
#     aapl_return = summary.loc[summary["symbol"] == "AAPL", "total_return"]
#     msft_return = summary.loc[summary["symbol"] == "MSFT", "total_return"]
#
#     # AAPL: (1100/1000) * (1050/1100) - 1 = approx 5%
#     expected_aapl = (1100 / 1000) * (1050 / 1100) - 1
#     # MSFT: (1010/1000) * (990/1010) - 1 = approx -1%
#     expected_msft = (1010 / 1000) * (990 / 1010) - 1
#
#     assert abs(aapl_return - expected_aapl).values[0] < 1e-4
#     assert abs(msft_return - expected_msft).values[0] < 1e-4
#
#     # Check that max drawdown is <= 0
#     assert summary.loc[summary["symbol"] == "AAPL", "max_drawdown"].values[0] <= 0
#     assert summary.loc[summary["symbol"] == "MSFT", "max_drawdown"].values[0] <= 0
#
#
# def test_max_drawdown_is_negative(sample_fills_portfolio):
#     analyzer = PerformanceAnalyzer(sample_fills_portfolio)
#     summary = analyzer.summary()
#
#     for symbol in summary.index:
#         dd = summary.loc[symbol, "max_drawdown"]
#         assert dd <= 0
