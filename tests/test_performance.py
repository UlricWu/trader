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
# def test_no_trades_performance():
#     from trader.performance import PerformanceAnalyzer
#     perf = PerformanceAnalyzer()
#     assert perf.calculate_total_return([]) == 0.0
#     assert perf.calculate_sharpe_ratio([]) == 0.0
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
