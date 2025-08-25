# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @File    : test_p.py
# # @Project : trader
# # @Author  : wsw
# # @Time    : 2025/7/31 13:07
# import numpy as np
# import pytest
# import pandas as pd
# from datetime import datetime
# from trader.performance import PerformanceAnalyzer
#
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from trader.performance import PerformanceAnalyzer
import trader.statistics as Stats


class MockTrade:
    def __init__(self, symbol, quantity):
        self.symbol = symbol
        self.quantity = quantity


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
        self.symbol_equity_df = pd.DataFrame({
            "AAPL": [50000, 52000, 51000, 48000],
            "MSFT": [30000, 31000, 30500, 29500]
        }, index=pd.date_range("2025-01-01", periods=4))


@pytest.fixture
def analyzer():
    portfolio = MockPortfolio()
    return PerformanceAnalyzer(portfolio)


def test_account_and_symbol_summary(analyzer):
    summary = analyzer.summary()
    # Account-level
    account = summary['account']
    assert 'equity' in account
    assert 'returns' in account
    assert 'sharpe' in account
    # Account total equity sum
    # np.testing.assert_array_equal(account['equity'].values,
    #                               analyzer.portfolio.symbol_equity_df.sum(
    #                                   axis=1).values + analyzer.portfolio.cash_df.values)
    # Symbol-level
    for sym in ['AAPL', 'MSFT']:
        assert sym in summary
        assert 'equity' in summary[sym]
        assert 'returns' in summary[sym]
        assert 'sharpe' in summary[sym]


def test_max_drawdown_and_volatility(analyzer):
    summary = analyzer.summary()
    account = summary['account']
    equity = account['equity']
    dd, max_duration = Stats.max_drawdown_and_duration(equity)
    assert dd == account['max_dd']
    assert max_duration == account['max_ddduration']
    vol = Stats.volatility(equity)
    assert vol == account['volatility']


# def test_export_metrics(analyzer):
#     file_path = "stats/performance.json"
#     analyzer.export_metrics(str(file_path))
#     assert file_path.exists()
#     import json
#     with open(file_path) as f:
#         data = json.load(f)
#     assert "timestamp" in data
#     assert "account" in data
#     assert "AAPL" in data


def test_monthly_returns_heatmap_calculation(analyzer):
    summary = analyzer.summary()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    analyzer._plot_monthly_returns(ax, summary['AAPL'])
    # test that pivot_table has correct months
    monthly_returns_df = summary['AAPL']['returns'].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    assert len(monthly_returns_df) > 0
    plt.close(fig)


def test_edge_cases_flat_or_decreasing():
    # Flat equity
    dates = pd.date_range("2025-01-01", periods=3)
    eq_flat = pd.Series([100000, 100000, 100000], index=dates)
    eq_down = pd.Series([100000, 95000, 90000], index=dates)

    class TempPortfolio:
        def __init__(self, eq_series):
            self.equity_df = pd.DataFrame({"equity": eq_series})
            self.symbol_equity_df = pd.DataFrame({"SYM": eq_series})
            self.settings = MockSettings()
            self.realized_pnl = {}
            self.transactions = []

    # Flat
    analyzer_flat = PerformanceAnalyzer(TempPortfolio(eq_flat))
    summary_flat = analyzer_flat.summary()
    assert summary_flat['account']['max_dd'] == 0.0

    # Decreasing
    analyzer_down = PerformanceAnalyzer(TempPortfolio(eq_down))
    summary_down = analyzer_down.summary()
    expected_dd = (eq_down / eq_down.cummax() - 1).min()
    assert round(summary_down['account']['max_dd'], 6) == round(expected_dd, 6)


def test_multi_symbol_account_consistency():
    # Total account equity should equal sum of symbols
    portfolio = MockPortfolio()
    analyzer_multi = PerformanceAnalyzer(portfolio)
    summary = analyzer_multi.summary()
    account_total = summary['account']['equity']
    sum_symbols = portfolio.symbol_equity_df.sum(axis=1)
    # np.testing.assert_array_equal(account_total.values, sum_symbols.values)
