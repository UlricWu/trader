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
        # self._stats: Dict[str, float] = {}

    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """
        Compute simple returns from price series.
        """
        return prices.pct_change().fillna(0)

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

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """
        Compute maximum drawdown from an equity curve.
        """
        cumulative_max = equity_curve.cummax()
        drawdowns = (equity_curve - cumulative_max) / cumulative_max
        return drawdowns.min()

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

        results = []
        # print(self.portfolio.equity_curve)
        for symbol, group in self.portfolio.equity_curve.groupby("symbol"):
            values = group["price"]

            total_return = self.total_return(group)

            returns = self.returns_daily(values)
            volatility = self.calculate_volatility(returns)
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            max_drawdown = self.calculate_max_drawdown(values)

            # Execute properties and collect results
            # result = {symbol: [(stat, getattr(self, stat)) for stat in stat_properties]}
            result = {
                "symbol": symbol,
                "total_return": total_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown
            }
            results.append(result)

        # Convert results into a Pandas DataFrame
        return pd.DataFrame(results)

    # @property
    def returns_daily(self, price: pd.Series):
        """Calculate daily returns based on price series"""
        return price.pct_change().dropna()

    def cumulative_returns_daily(self, returns_daily: pd.Series) -> float:
        """Calculate cumulative returns from daily returns"""
        return (1 + returns_daily).cumprod() - 1

    def total_return(self, equity_curve) -> float:
        price = equity_curve['price']
        return price.tail(1) / price.head(1) - 1

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

    #
    # @property
    # def cumulative_returns(self):
    #     """Calculate cumulative returns for the portfolio"""
    #     portfolio_returns = self._calculate_returns(self.portfolio.history["cash_balance"])
    #     self.performance_metrics["portfolio_cumulative_returns"] = self._calculate_cumulative_returns(portfolio_returns)
    #
    #     if self.benchmark is not None:
    #         benchmark_returns = self._calculate_returns(self.benchmark)
    #         self.performance_metrics["benchmark_cumulative_returns"] = self._calculate_cumulative_returns(
    #             benchmark_returns)
    #
    #
    #
    # # def analyze(self) -> Dict[str, float]:
    # #     self._stats['total_return'] = self._total_return()
    # #     self._stats['annualized_return'] = self._annualized_return()
    # #     self._stats['volatility'] = self._annualized_volatility()
    # #     self._stats['sharpe_ratio'] = self._sharpe_ratio()
    # #     self._stats['max_drawdown'] = self._max_drawdown()
    # #     self._stats['win_rate'] = self._win_rate()
    # #     self._stats['avg_trade_return'] = self._avg_trade_return()
    # #     self._stats['profit_factor'] = self._profit_factor()
    # #     return self._stats
    #

    # #
    # # def _annualized_return(self) -> float:
    # #     n_days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
    # #     return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) ** (252 / n_days) - 1
    # #
    # # def _annualized_volatility(self) -> float:
    # #     return self.returns.std() * np.sqrt(252)
    # #
    # # def _sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
    # #     excess_return = self._annualized_return() - risk_free_rate
    # #     volatility = self._annualized_volatility()
    # #     return excess_return / volatility if volatility > 0 else np.nan
    # #
    # # def _max_drawdown(self) -> float:
    # #     cum_max = self.equity_curve.cummax()
    # #     drawdown = self.equity_curve / cum_max - 1
    # #     return drawdown.min()
    # #
    # # def _win_rate(self) -> float:
    # #     wins = (self.returns > 0).sum()
    # #     total = self.returns.count()
    # #     return wins / total if total > 0 else np.nan
    # #
    # # def _avg_trade_return(self) -> float:
    # #     return self.returns.mean()
    # #
    # # def _profit_factor(self) -> float:
    # #     gross_profit = self.returns[self.returns > 0].sum()
    # #     gross_loss = -self.returns[self.returns < 0].sum()
    # #     return gross_profit / gross_loss if gross_loss > 0 else np.nan
    # #
    # # def plot_equity_curve(self):
    # #     dates = [snap.timestamp for snap in self.history]
    # #     plt.figure(figsize=(10, 5))
    # #     plt.plot(dates, self.equity_curve, label="Equity Curve")
    # #     plt.title("Equity Curve")
    # #     plt.xlabel("Date")
    # #     plt.ylabel("Portfolio Value")
    # #     plt.legend()
    # #     plt.grid(True)
    # #     plt.tight_layout()
    # #     plt.show()
    # #
    # # @property
    # # def equity_curve(self) -> List[float]:
    # #     return pd.DataFrame([snap.value for snap in self.history])
    # #
    # # @property
    # # def returns(self):
    # #     # curve = self.equity_curve
    # #
    # #     return self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1
    # #     # print(curve)
    # #     # return np.diff(curve) / curve[:-1]
    # #
    # # @property
    # # def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
    # #     print(self.returns)
    # #     excess_returns = self.returns - risk_free_rate / 252
    # #     return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    # #
    # # @property
    # # def max_drawdown(self) -> float:
    # #     cumulative = np.array(self.equity_curve)
    # #     peak = np.maximum.accumulate(cumulative)
    # #     drawdown = (cumulative - peak) / peak
    # #     return np.min(drawdown)
    # #
    # # @property
    # # def start_date(self):
    # #     # return self.trades['trade_date'].head(1).iloc[0]
    # #     return self.history[0].timestamp
    # #
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
    # #
    # # @property
    # # def avg_return_d(self):
    # #     simple_return = 'simple_return'
    # #     if simple_return not in self.trades.columns.values:
    # #         self.calculate_change_pct()
    # #
    # #     return self.trades[simple_return].mean()
    # #
    # # #
    # # def calculate_change_pct(self):
    # #     self.trades['simple_return'] = self.trades['close'].pct_change()
    # #
    # # @property
    # # def avg_return(self, days=None):
    # #     if not days:
    # #         days = len(self.trades)
    # #     return self.avg_return_d * days
    # #
    # # @property
    # # def log_return(self, days=None):
    # #     if not days:
    # #         days = self.days
    # #     log_return = np.log(self.trades['close'] / self.trades['close'].shift(1))
    # #     return log_return.mean() * days
    # #
    # # @property
    # # def sharpe_ratio(self, risk_free_rate=None, days=None):
    # #     if not days:
    # #         days = self.days
    # #     if not risk_free_rate:
    # #         risk_free_rate = self.risk_free_rate
    # #     returns = self.trades['simple_return']
    # #     return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(days)
