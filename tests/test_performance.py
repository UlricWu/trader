#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : test_p.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/7/31 13:07
import numpy as np
import pytest
import pandas as pd
from datetime import datetime
from trader.performance import PerformanceAnalyzer


class MockSettings:
    class trading:
        INITIAL_CASH = 100000.0


class MockPortfolio:
    def __init__(self):
        self.settings = MockSettings()
        self.realized_pnl = {"AAPL": 150.0, "MSFT": -50.0}
        self.transactions = [
            MockTrade("AAPL", 10), MockTrade("AAPL", 20),
            MockTrade("MSFT", 15)
        ]
        self.history = [
            (datetime(2025, 1, 1), 100000),
            (datetime(2025, 1, 2), 101000),
            (datetime(2025, 1, 3), 101200),
            (datetime(2025, 1, 4), 100800)
        ]
        self.equity_df = pd.DataFrame(self.history, columns=["datetime", "equity"]).set_index("datetime")
        self.symbol_equity_history = {
            "AAPL": [
                (datetime(2025, 1, 1), 50000),
                (datetime(2025, 1, 2), 52000),
                (datetime(2025, 1, 3), 51000),
                (datetime(2025, 1, 4), 48000)
            ],
            "MSFT": [
                (datetime(2025, 1, 1), 30000),
                (datetime(2025, 1, 2), 31000),
                (datetime(2025, 1, 3), 30500),
                (datetime(2025, 1, 4), 29500)
            ]
        }

    @property
    def symbol_equity_dfs(self):
        return {
            symbol: pd.DataFrame(history, columns=["datetime", "equity"]).set_index("datetime")
            for symbol, history in self.symbol_equity_history.items()
        }


class MockTrade:
    def __init__(self, symbol, quantity):
        self.symbol = symbol
        self.quantity = quantity


@pytest.fixture
def analyzer():
    portfolio = MockPortfolio()
    return PerformanceAnalyzer(portfolio)


def test_account_summary(analyzer):
    summary = analyzer.account_summary()
    assert round(summary["total_return"], 6) == 0.008
    assert round(summary["max_drawdown"], 6) == -0.003953
    assert summary["initial_cash"] == 100000.0
    assert summary["final_equity"] == 100800
    assert summary["total_realized_pnl"] == 100.0


def test_symbol_summary(analyzer):
    symbol_stats = analyzer.symbol_summary()
    assert symbol_stats["AAPL"]["realized_pnl"] == 150.0
    assert symbol_stats["AAPL"]["trade_count"] == 2
    assert symbol_stats["AAPL"]["total_volume"] == 30
    assert round(symbol_stats["AAPL"]["max_drawdown"], 6) == -0.076923

    assert symbol_stats["MSFT"]["realized_pnl"] == -50.0
    assert symbol_stats["MSFT"]["trade_count"] == 1
    assert symbol_stats["MSFT"]["total_volume"] == 15
    assert round(symbol_stats["MSFT"]["max_drawdown"], 6) == -0.048387

    ann_return = analyzer.annualized_return("AAPL")

    total_return = (4800 - 5000) / 5000
    duration_days = 4
    annual_factor = 252 / duration_days
    expected = (1 + total_return) ** annual_factor - 1
    assert abs(ann_return - round(expected, 6)) < 1e-5


def test_full_report_structure(analyzer):
    report = analyzer.summary()
    assert "account" in report
    assert "per_symbol" in report
    assert "AAPL" in report["per_symbol"]
    assert isinstance(report["account"], dict)
    assert isinstance(report["per_symbol"], dict)


def test_max_drawdown_calculation():
    df = pd.DataFrame([
        ["2025-01-01", 100000],
        ["2025-01-02", 105000],
        ["2025-01-03", 103000],
        ["2025-01-04", 98000],
        ["2025-01-05", 99000],
        ["2025-01-06", 107000]
    ], columns=["datetime", "equity"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    class TempPortfolio:
        def __init__(self, equity_df):
            self.equity_df = equity_df
            self.realized_pnl = {}
            self.settings = MockSettings()
            self.transactions = []
            self.history = list(equity_df.reset_index().itertuples(index=False, name=None))

    temp_portfolio = TempPortfolio(df)
    analyzer = PerformanceAnalyzer(temp_portfolio)
    drawdown = analyzer.compute_max_drawdown()
    assert round(drawdown, 4) == -0.0667

    returns = df["equity"].pct_change().dropna()
    volatility = returns.std()
    sharpe = returns.mean() / volatility * np.sqrt(252)

    # assert round(volatility, 6) > 0
    print(analyzer.sharpe_ratio())
    assert round(sharpe, 4) == analyzer.sharpe_ratio()


def test_max_drawdown_flat_equity():
    df = pd.DataFrame([
        ["2025-01-01", 100000],
        ["2025-01-02", 100000],
        ["2025-01-03", 100000]
    ], columns=["datetime", "equity"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    class TempPortfolio:
        def __init__(self, equity_df):
            self.equity_df = equity_df
            self.realized_pnl = {}
            self.settings = MockSettings()
            self.transactions = []
            self.history = list(equity_df.reset_index().itertuples(index=False, name=None))

    temp_portfolio = TempPortfolio(df)
    analyzer = PerformanceAnalyzer(temp_portfolio)
    drawdown = analyzer.compute_max_drawdown()
    assert drawdown == 0.0


def test_max_drawdown_only_down():
    df = pd.DataFrame([
        ["2025-01-01", 100000],
        ["2025-01-02", 95000],
        ["2025-01-03", 90000]
    ], columns=["datetime", "equity"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    class TempPortfolio:
        def __init__(self, equity_df):
            self.equity_df = equity_df
            self.realized_pnl = {}
            self.settings = MockSettings()
            self.transactions = []
            self.history = list(equity_df.reset_index().itertuples(index=False, name=None))

    temp_portfolio = TempPortfolio(df)
    analyzer = PerformanceAnalyzer(temp_portfolio)
    drawdown = analyzer.compute_max_drawdown()
    assert round(drawdown, 4) == -0.1


def test_rolling_max_drawdown():
    dates = pd.date_range(start="2025-01-01", periods=10, freq="D")
    equity = [100000, 102000, 101000, 99000, 98000, 100000, 101000, 103000, 104000, 105000]
    df = pd.DataFrame({"datetime": dates, "equity": equity}).set_index("datetime")

    class TempPortfolio:
        def __init__(self, equity_df):
            self.equity_df = equity_df
            self.realized_pnl = {}
            self.settings = MockSettings()
            self.transactions = []
            self.history = list(equity_df.reset_index().itertuples(index=False, name=None))

    temp_portfolio = TempPortfolio(df)
    analyzer = PerformanceAnalyzer(temp_portfolio)

    rolling_max = df["equity"].cummax()
    rolling_drawdown = df["equity"] / rolling_max - 1.0
    max_drawdown = rolling_drawdown.min()

    assert round(max_drawdown, 6) == analyzer.compute_max_drawdown()


def test_yearly_return():
    df = pd.DataFrame([
        ["2024-01-01", 100000],
        ["2024-12-31", 105000],
        ["2025-01-01", 110000],
    ], columns=["datetime", "equity"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)

    class TempPortfolio:
        def __init__(self, equity_df):
            self.equity_df = equity_df
            self.realized_pnl = {}
            self.settings = MockSettings()
            self.transactions = []
            self.history = list(equity_df.reset_index().itertuples(index=False, name=None))

    temp_portfolio = TempPortfolio(df)
    analyzer = PerformanceAnalyzer(temp_portfolio)

    # Calculate expected annualized return manually:
    # total return = (110000-100000)/100000 = 0.1
    # days = 2 (Jan 4 - Jan 2)
    # annual_factor = 252 / 2
    # annualized_return = (1 + 0.1)^126 - 1
    expected = (1 + 0.1) ** (252 / 3) - 1
    # assert analyzer.compute_annualized_return() == expected
    assert abs(analyzer.annualized_return() - round(expected, 6)) < 1e-5
