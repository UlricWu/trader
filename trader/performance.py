#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : performance.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/22 10:46

from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

#
# class PerformanceAnalyzer:
#     days = 252
#
#     def __init__(self, equity_df: pd.DataFrame):
#         self.df = equity_df.copy()
#
#     def calculate_return(self, dataframe):
#         return dataframe.pct_change().fillna(0)
#
#     def calcuate_cumulative_return(self, returns):
#         return (1 + returns).cumprod()
#
#     def calculate_drawdown(self, cumulative):
#         return cumulative / cumulative.cummax() - 1
#
#     def calculate_total_return(self, cumulative):
#         return cumulative.iloc[-1] - 1
#
#     def calculate_sharpe_ratio(self, returns, risk_free_rate=0.01) -> float:
#         excess = returns - risk_free_rate / self.days
#         return np.sqrt(self.days) * excess.mean() / excess.std()
#
#     def calculate_annual_volatility(self, returns):
#         return returns.std() * np.sqrt(self.days)
#
#     def calculate_annual_return(self, returns):
#         return returns.mean() * self.days
#
#     def stats(self) -> dict:
#         returns = self.calculate_return(self.df["equity"])
#
#         total_return = self.df["cumulative"].iloc[-1] - 1
#         sharpe_ratio = self._sharpe_ratio(self.df["returns"])
#         max_drawdown = self.df["drawdown"].min()
#         annual_volatility = self.df["returns"].std() * np.sqrt(252)
#         annual_return = self.df["returns"].mean() * 252
#
#         return {
#             "Total Return": f"{total_return:.2%}",
#             "Annual Return": f"{annual_return:.2%}",
#             "Annual Volatility": f"{annual_volatility:.2%}",
#             "Sharpe Ratio": f"{sharpe_ratio:.2f}",
#             "Max Drawdown": f"{max_drawdown:.2%}",
#         }
#
#     def plot(self) -> None:
#         fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
#
#         self.df["cumulative"].plot(ax=ax[0], title="Cumulative Returns", color="blue")
#         ax[0].set_ylabel("Cumulative Return")
#
#         self.df["drawdown"].plot(ax=ax[1], title="Drawdown", color="red")
#         ax[1].set_ylabel("Drawdown")
#         ax[1].set_xlabel("Date")
#
#         plt.tight_layout()
#         plt.show()
#
#     def summary(self) -> None:
#         print("Performance Summary")
#         print("=" * 30)
#         for key, value in self.stats().items():
#             print(f"{key:<20}: {value}")
