#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : metrics.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/3/12 14:27
import sys

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List

from trader.portfolio import Portfolio


class PositionSizer:
    def __init__(self, risk_percentage: float = 0.01):
        """Set the risk percentage of equity per trade."""
        self.risk_percentage = risk_percentage

    def calculate(self, df: pd.DataFrame, symbol: str, equity: float) -> int:
        """Calculate the number of shares to buy based on equity and the current price."""
        price = df.iloc[-1]["close"]
        position_size = int((equity * self.risk_percentage) // price)  # Allocate a fixed percentage of equity
        return position_size

    # !quant_trading_system/performance/analyzer.py

    from typing import List, Dict
    import numpy as np
    import pandas as pd


class PerformanceAnalyzer:
    days = 252
    risk_free_rate = 0.02

    def __init__(self, portfolio: "Portfolio", benchmark: pd.Series = None):
        """
        Parameters:
            equity_curve: A Pandas Series of daily portfolio values (indexed by datetime)
        """
        self.portfolio = portfolio
        self.benchmark = benchmark
        # self._stats: Dict[str, float] = {}

    @property
    def returns_daily(self):
        """Calculate daily returns based on price series"""
        return self.portfolio.history.pct_change().dropna()

    @property
    def cumulative_returns_daily(self):
        """Calculate cumulative returns from daily returns"""
        return (1 + self.returns_daily).cumprod() - 1

    @property
    def cumulative_returns(self):
        """Calculate cumulative returns for the portfolio"""
        portfolio_returns = self._calculate_returns(self.portfolio.history["cash_balance"])
        self.performance_metrics["portfolio_cumulative_returns"] = self._calculate_cumulative_returns(portfolio_returns)

        if self.benchmark is not None:
            benchmark_returns = self._calculate_returns(self.benchmark)
            self.performance_metrics["benchmark_cumulative_returns"] = self._calculate_cumulative_returns(
                benchmark_returns)

    def summary(self):
        """Calls all stat functions and returns a DataFrame."""
        stat_properties = [
            "start_date",
            "end_date",
            "period",
            # "start_value",
            # "end_value",
            # "equity_curve",
            "avg_return_d",
            # "avg_return",
            # "log_return",
            "sharpe_ratio",
            "max_drawdown"
        ]

        # Execute properties and collect results
        results = [(stat, getattr(self, stat)) for stat in stat_properties]

        # Convert results into a Pandas DataFrame
        return pd.DataFrame(results, columns=["Stat Name", "Stat Value"])

    # def analyze(self) -> Dict[str, float]:
    #     self._stats['total_return'] = self._total_return()
    #     self._stats['annualized_return'] = self._annualized_return()
    #     self._stats['volatility'] = self._annualized_volatility()
    #     self._stats['sharpe_ratio'] = self._sharpe_ratio()
    #     self._stats['max_drawdown'] = self._max_drawdown()
    #     self._stats['win_rate'] = self._win_rate()
    #     self._stats['avg_trade_return'] = self._avg_trade_return()
    #     self._stats['profit_factor'] = self._profit_factor()
    #     return self._stats

    # def _total_return(self) -> float:
    #     return self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
    #
    # def _annualized_return(self) -> float:
    #     n_days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
    #     return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) ** (252 / n_days) - 1
    #
    # def _annualized_volatility(self) -> float:
    #     return self.returns.std() * np.sqrt(252)
    #
    # def _sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
    #     excess_return = self._annualized_return() - risk_free_rate
    #     volatility = self._annualized_volatility()
    #     return excess_return / volatility if volatility > 0 else np.nan
    #
    # def _max_drawdown(self) -> float:
    #     cum_max = self.equity_curve.cummax()
    #     drawdown = self.equity_curve / cum_max - 1
    #     return drawdown.min()
    #
    # def _win_rate(self) -> float:
    #     wins = (self.returns > 0).sum()
    #     total = self.returns.count()
    #     return wins / total if total > 0 else np.nan
    #
    # def _avg_trade_return(self) -> float:
    #     return self.returns.mean()
    #
    # def _profit_factor(self) -> float:
    #     gross_profit = self.returns[self.returns > 0].sum()
    #     gross_loss = -self.returns[self.returns < 0].sum()
    #     return gross_profit / gross_loss if gross_loss > 0 else np.nan
    #
    # def plot_equity_curve(self):
    #     dates = [snap.timestamp for snap in self.history]
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(dates, self.equity_curve, label="Equity Curve")
    #     plt.title("Equity Curve")
    #     plt.xlabel("Date")
    #     plt.ylabel("Portfolio Value")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()
    #
    # @property
    # def equity_curve(self) -> List[float]:
    #     return pd.DataFrame([snap.value for snap in self.history])
    #
    # @property
    # def returns(self):
    #     # curve = self.equity_curve
    #
    #     return self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
    #     # print(curve)
    #     # return np.diff(curve) / curve[:-1]
    #
    # @property
    # def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
    #     print(self.returns)
    #     excess_returns = self.returns - risk_free_rate / 252
    #     return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    #
    # @property
    # def max_drawdown(self) -> float:
    #     cumulative = np.array(self.equity_curve)
    #     peak = np.maximum.accumulate(cumulative)
    #     drawdown = (cumulative - peak) / peak
    #     return np.min(drawdown)
    #
    # @property
    # def start_date(self):
    #     # return self.trades['trade_date'].head(1).iloc[0]
    #     return self.history[0].timestamp
    #
    # @property
    # def end_date(self):
    #     return self.history[-1].timestamp
    #
    # @property
    # def period(self):
    #
    #     return (self.end_date - self.start_date).days
    #
    # @property
    # def start_value(self):
    #     # return self.trades['close'].head(1).iloc[0]
    #     return self.history[0].value
    #
    # @property
    # def end_value(self):
    #     return self.history[-1].value
    #
    # @property
    # def avg_return_d(self):
    #     simple_return = 'simple_return'
    #     if simple_return not in self.trades.columns.values:
    #         self.calculate_change_pct()
    #
    #     return self.trades[simple_return].mean()
    #
    # #
    # def calculate_change_pct(self):
    #     self.trades['simple_return'] = self.trades['close'].pct_change()
    #
    # @property
    # def avg_return(self, days=None):
    #     if not days:
    #         days = len(self.trades)
    #     return self.avg_return_d * days
    #
    # @property
    # def log_return(self, days=None):
    #     if not days:
    #         days = self.days
    #     log_return = np.log(self.trades['close'] / self.trades['close'].shift(1))
    #     return log_return.mean() * days
    #
    # @property
    # def sharpe_ratio(self, risk_free_rate=None, days=None):
    #     if not days:
    #         days = self.days
    #     if not risk_free_rate:
    #         risk_free_rate = self.risk_free_rate
    #     returns = self.trades['simple_return']
    #     return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(days)
