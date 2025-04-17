#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : metrics.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/3/12 14:27
import numpy as np
import pandas as pd

from trader.portfolio import Portfolio


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

    def summary(self):
        """Calls all stat functions and returns a DataFrame."""
        # stat_properties = [
        #     "start_date",
        #     "end_date",
        #     "period",
        #     # "start_value",
        #     # "end_value",
        #     # "equity_curve",
        #     "avg_return_d",
        #     # "avg_return",
        #     # "log_return",
        #     "sharpe_ratio",
        #     "max_drawdown"
        # ]

        results = []
        for symbol, group in self.portfolio.equity_curve.groupby("symbol"):
            values = group["price"]
            returns = self.calculate_returns(values)
            total_return = self.total_return(returns)
            volatility = self.calculate_volatility(returns)
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            max_drawdown = self.calculate_max_drawdown(values)
            win_rate = self.win_rate(returns)
            profit_factor = self.profit_factor(returns)

            # Execute properties and collect results
            result = {
                "symbol": symbol,
                "total_return": total_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "win_rate": win_rate,
                "profit_factor": profit_factor
            }
            results.append(result)

        # Convert results into a Pandas DataFrame
        return pd.DataFrame(results)

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

    # def _annualized_return(self) -> float:
    #     n_days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
    #     return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) ** (252 / n_days) - 1

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """
        Compute maximum drawdown from an equity curve.
        """
        cumulative_max = equity_curve.cummax()
        drawdowns = (equity_curve - cumulative_max) / cumulative_max
        return drawdowns.min()

    def calculate_volatility(self, returns: pd.Series, day=None) -> float:
        if not day:
            day = self.days
        return returns.std() * np.sqrt(day)

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = None, days: float = None) -> float:
        if not risk_free_rate:
            risk_free_rate = self.risk_free_rate
        if not days:
            days = self.days
        excess_returns = returns - risk_free_rate / days

        volatility = self.calculate_volatility(returns)
        return excess_returns.mean() / volatility if volatility != 0 else np.nan

    def calculate_max_drawdown(self, values: pd.Series) -> float:
        cumulative = values.cummax()
        drawdown = (values - cumulative) / cumulative
        return drawdown.min()

    def win_rate(self, returns) -> float:
        wins = (returns > 0).sum()
        total = returns.count()
        return wins / total if total > 0 else np.nan

    def profit_factor(self, returns) -> float:
        gross_profit = returns[returns > 0].sum()
        gross_loss = -returns[returns < 0].sum()
        return gross_profit / gross_loss if gross_loss > 0 else np.nan

    # def start_date(self):
    #     # return self.trades['trade_date'].head(1).iloc[0]
    #     return self.history[0].timestamp
    # # #
    # # @property
    # # def end_date(self):
    # #     return self.history[-1].timestamp
    # #
    # # @property
    # # def period(self):
    # #
    # #     return (self.end_date - self.start_date).days
    # #
    # # @property
    # # def start_value(self):
    # #     # return self.trades['close'].head(1).iloc[0]
    # #     return self.history[0].value
    # #
    # # @property
    # # def end_value(self):
    # #     return self.history[-1].value
