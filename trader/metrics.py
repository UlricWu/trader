#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : metrics.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/3/12 14:27
from typing import List

import numpy as np
import pandas as pd

from trader.portfolio import Portfolio


class PerformanceAnalyzer:
    days = 252
    risk_free_rate = 0.02

    # define your default metric list (public names)
    DEFAULT_METRICS = [
        "total_return",
        "annualized_return",
        "volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor"
    ]

    def __init__(self, portfolio: "Portfolio", benchmark: pd.Series = None):
        """
        Parameters:
            equity_curve: A Pandas Series of daily portfolio values (indexed by datetime)
        """
        self.portfolio = portfolio
        self.benchmark = benchmark

    def summary(self, metrics: List[str] = None) -> pd.DataFrame:
        """
        Compute all requested metrics for each symbol and return a DataFrame.

        :param metrics: list of metric names; must match one of the public
                        methods on this class.  If None, uses DEFAULT_METRICS.
        """
        metrics = metrics or self.DEFAULT_METRICS
        results = []
        for symbol, group in self.portfolio.equity_curve.groupby("symbol"):
            values = group["price"]
            returns = self.calculate_returns(values)

            start_day = group["date"].head(1).values[0]
            end_day = group["date"].tail(1).values[0]
            row = {"symbol": symbol,
                   "start_day": start_day,
                   "end_date": end_day,
                   "period": end_day - start_day,
                   "start_value": values.head(1).values[0],
                   "end_value": values.tail(1).values[0]

                   }
            for m in metrics:
                if not hasattr(self, m):
                    raise ValueError(f"Metric '{m}' not found on PerformanceAnalyzer")
                func = getattr(self, m)
                # choose argument based on signature
                if m == "max_drawdown":
                    row[m] = func(values)
                else:
                    row[m] = func(returns)

            results.append(row)

        # Convert results into a Pandas DataFrame
        return pd.DataFrame(results).set_index("symbol")

    # --- public wrappers around your static calculations ---
    def total_return(self, returns: pd.DataFrame) -> float:
        """Overall % change over the period"""
        return (1 + returns).prod() - 1

    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """
        Compute daily simple returns from price series.
        """

        return prices.pct_change().fillna(0)

    def log_return(self, returns, days=None):
        if not days:
            days = self.days
        log_return = np.log(returns)
        return log_return.mean() * days

    @staticmethod
    def equity_curve(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
        """
        Build equity curve from returns, starting at initial_value.
        """
        return (1 + returns).cumprod() * initial_value

    @staticmethod
    def avg_return(returns: pd.Series) -> float:
        """
        Compute average (mean) return.
        """
        return returns.mean()

    def annualized_return(self, returns: pd.DataFrame) -> float:
        total_ret = self.total_return(returns)
        periods = len(returns.dropna())
        # assume daily returns; 252 trading days
        return (1 + total_ret) ** (self.days / periods) - 1

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """
        Compute maximum drawdown from an equity curve.
        """
        cumulative_max = equity_curve.cummax()
        drawdowns = (equity_curve - cumulative_max) / cumulative_max
        return drawdowns.min()

    def volatility(self, returns: pd.Series, day=None) -> float:
        if not day:
            day = self.days
        return returns.std() * np.sqrt(day)

    def sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = None, days: float = None) -> float:
        if not risk_free_rate:
            risk_free_rate = self.risk_free_rate
        if not days:
            days = self.days
        excess_returns = returns - risk_free_rate / days

        volatility = self.volatility(returns)
        return excess_returns.mean() / volatility if volatility != 0 else np.nan

    def win_rate(self, returns) -> float:
        wins = (returns > 0).sum()
        total = returns.count()
        return wins / total if total > 0 else np.nan

    def profit_factor(self, returns) -> float:
        gross_profit = returns[returns > 0].sum()
        gross_loss = -returns[returns < 0].sum()
        return gross_profit / gross_loss if gross_loss > 0 else np.nan
