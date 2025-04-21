# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @File    : test_prof.py
# # @Project : trader
# # @Author  : wsw
# # @Time    : 2025/4/10 17:16
#
# import pytest
# from datetime import datetime
# from trader.portfolio import Portfolio, Position, PortfolioSnapshot, Fill
# from collections import defaultdict
#
# # Test fixture to set up a portfolio with initial positions
# approx = pytest.approx(153.33, rel=1e-2)
#
#
# @pytest.fixture
# def setup_portfolio():
#     portfolio = Portfolio(initial_cash=100000)
#     portfolio.positions = {
#         "AAPL": Position(symbol="AAPL", quantity=100, entry_price=150),
#         "MSFT": Position(symbol="MSFT", quantity=50, entry_price=250),
#     }
#     return portfolio
#
#
# # Test: Portfolio initialization with correct cash and positions
# def test_initialization(setup_portfolio):
#     portfolio = setup_portfolio
#     assert portfolio.cash == 100000, "Initial cash is incorrect"
#     assert len(portfolio.positions) == 2, "Incorrect number of positions"
#
#
# # Test: Add a position to the portfolio
# def test_add_position(setup_portfolio):
#     portfolio = setup_portfolio
#     portfolio.positions["GOOG"] = Position(symbol="GOOG", quantity=10, entry_price=1000)
#     assert "GOOG" in portfolio.positions, "GOOG position was not added"
#     assert portfolio.positions["GOOG"].quantity == 10, "GOOG position quantity is incorrect"
#
#
# # Test: Update position with a fill
# def test_update_position(setup_portfolio):
#     portfolio = setup_portfolio
#     fill = Fill(symbol="AAPL", quantity=50, price=160, date=datetime.now())
#     portfolio.update([fill])
#
#     # price = (150*100+50*160)/(100+50) = 153.33
#     # print(price)
#
#     position = portfolio.positions["AAPL"]
#     assert position.quantity == 150, f"Position quantity after fill is incorrect: {position.quantity}"
#     assert position.entry_price == approx, f"Position average price after fill is incorrect: {position.entry_price}"
#
#
# # Test: Calculate market value based on positions and prices
# def test_market_value(setup_portfolio):
#     portfolio = setup_portfolio
#     mock_prices = {"AAPL": 155, "MSFT": 260}
#
#     market_value = portfolio.market_value(mock_prices)
#     expected_market_value = (100 * 155) + (50 * 260)  # 15500 + 13000 = 28500
#
#     assert market_value == expected_market_value, f"Market value is incorrect: {market_value}"
#
#
# # Test: Take a snapshot with mock prices
# def test_take_snapshot(setup_portfolio):
#     portfolio = setup_portfolio
#     mock_prices = {"AAPL": 155, "MSFT": 260}
#
#     # Take snapshot
#     portfolio.take_snapshot(prices=mock_prices)
#
#     assert len(portfolio.snapshots) > 0, "No snapshots recorded"
#     snapshot = portfolio.snapshots[-1]
#
#     # Validate snapshot values
#     market_value = (100 * 155) + (50 * 260)  # 15500 + 13000 = 28500
#     total_value = portfolio.cash + market_value
#
#     assert snapshot.cash == portfolio.cash, f"Cash in snapshot is incorrect: {snapshot.cash}"
#     assert snapshot.market_value == market_value, f"Market value in snapshot is incorrect: {snapshot.market_value}"
#     assert snapshot.total_value == total_value, f"Total value in snapshot is incorrect: {snapshot.total_value}"
#     assert isinstance(snapshot.timestamp, datetime), "Timestamp in snapshot is incorrect"
#
#
# # Test: Adding two positions using the overloaded __add__ method
# def test_add_positions():
#     position1 = Position(symbol="AAPL", quantity=100, entry_price=150)
#     position2 = Position(symbol="AAPL", quantity=50, entry_price=160)
#
#     new_position = position1 + position2
#     assert new_position.quantity == 150, "Total quantity after addition is incorrect"
#     assert new_position.entry_price == approx, "Average price after addition is incorrect"
#
#
# # Test: Correctly handling fills for different symbols
# def test_fill_position():
#     position = Position(symbol="AAPL", quantity=100, entry_price=150)
#     fill = Fill(symbol="AAPL", quantity=50, price=160, date=datetime.now())
#     position = position + fill  # Using the overloaded __add__ method
#
#     assert position.quantity == 150, f"Position quantity after fill is incorrect: {position.quantity}"
#     assert position.entry_price == approx, f"Position average price after fill is incorrect: {position.entry_price}"
#
#
# # Test: PortfolioSnapshot creation
# def test_portfolio_snapshot():
#     portfolio = Portfolio(initial_cash=100000)
#     portfolio.positions = {
#         "AAPL": Position(symbol="AAPL", quantity=100, entry_price=150),
#         "MSFT": Position(symbol="MSFT", quantity=50, entry_price=250),
#     }
#     mock_prices = {"AAPL": 155, "MSFT": 260}
#
#     # Take snapshot
#     portfolio.take_snapshot(prices=mock_prices)
#     snapshot = portfolio.snapshots[-1]
#
#     # Validate that the snapshot contains the correct values
#     assert snapshot.cash == portfolio.cash, f"Snapshot cash is incorrect: {snapshot.cash}"
#     assert snapshot.market_value == (100 * 155) + (50 * 260), f"Snapshot market value is incorrect"
#     assert snapshot.total_value == portfolio.cash + snapshot.market_value, f"Snapshot total value is incorrect"
#     assert isinstance(snapshot.timestamp, datetime), "Timestamp is not a datetime"
#
#
# # Test: Handling missing prices for market value calculation
# def test_missing_prices():
#     portfolio = Portfolio(initial_cash=100000)
#     portfolio.positions = {
#         "AAPL": Position(symbol="AAPL", quantity=100, entry_price=150),
#     }
#     mock_prices = {}  # Missing price for AAPL but used current value
#
#     # Portfolio should handle missing prices gracefully and calculate market value as 0
#     market_value = portfolio.market_value(mock_prices)
#     assert market_value == 15000, f"Market value should be 0 but is {market_value}"
#
#
# # Test: Portfolio snapshot with missing prices
# def test_take_snapshot_missing_prices():
#     portfolio = Portfolio(initial_cash=100000)
#     portfolio.positions = {
#         "AAPL": Position(symbol="AAPL", quantity=100, entry_price=150),
#     }
#     mock_prices = {}  # Missing price for AAPL but used current value
#
#     portfolio.take_snapshot(prices=mock_prices)
#     snapshot = portfolio.snapshots[-1]
#
#     # The market value should be 0 since the price is missing
#     assert snapshot.market_value == 15000, f"Market value in snapshot should be 0 but is {snapshot.market_value}"
