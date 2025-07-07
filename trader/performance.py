#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : performance.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/22 10:46
import os
from datetime import datetime
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from trader.portfolio import Portfolio

import numpy as np
from typing import List


class Metrics:

    @staticmethod
    def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        excess_returns = np.array(returns) - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(252) if len(returns) > 1 else 0.0

    @staticmethod
    def sortino_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        excess_returns = np.array(returns) - risk_free_rate
        downside = excess_returns[excess_returns < 0]
        downside_std = np.std(downside, ddof=1)
        return np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std != 0 else 0.0

    @staticmethod
    def max_drawdown(equity_curve: List[float]) -> float:
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()

    @staticmethod
    def volatility(returns: List[float]) -> float:
        return np.std(returns, ddof=1) * np.sqrt(252)

    @staticmethod
    def total_return(equity_curve: List[float]) -> float:
        return (equity_curve[-1] / equity_curve[0]) - 1.0 if equity_curve else 0.0

    @staticmethod
    def annual_return(equity_curve: List[float], days: int = 252) -> float:
        return ((equity_curve[-1] / equity_curve[0]) ** (252 / days)) - 1.0 if equity_curve else 0.0


class PerformanceAnalyzer:
    days = 252

    # define your default metric list (public names)
    DEFAULT_METRICS = [
        "initial_cash",
        "final_equity",
        "total_return",
        "num_trades",
        "win_rate",
        "average_win",
        "average_loss",
        "profit_factor"
    ]

    def __init__(self, portfolio: "Portfolio", benchmark: pd.Series = None, risk_free_rate: float = 0.0):
        """
        Parameters:
            equity_curve: A Pandas Series of daily portfolio values (indexed by datetime)
        """
        self.portfolio = portfolio
        self.benchmark = benchmark

        self.snapshots = portfolio.daily_snapshots
        self.transactions = portfolio.transactions
        self.risk_free_rate = risk_free_rate

        self.equity_curve = portfolio.equity_curve
        # self.returns = np.diff(self.equity_curve) / self.equity_curve[:-1] if len(self.equity_curve) > 1 else [0.0]

    # def summary(self) -> dict:
    #     return {
    #         "total_return": Metrics.total_return(self.equity_curve),
    #         "annual_return": Metrics.annual_return(self.equity_curve),
    #         "max_drawdown": Metrics.max_drawdown(self.equity_curve),
    #         "volatility": Metrics.volatility(self.returns),
    #         "sharpe_ratio": Metrics.sharpe_ratio(self.returns, self.risk_free_rate),
    #         "sortino_ratio": Metrics.sortino_ratio(self.returns, self.risk_free_rate),
    #         "win_rate": Metrics.calculate_win_rate(),
    #         "profit_factor": Metrics.calculate_profit_factor(),
    #     }

    def _analyze_symbol(self, symbol: str, pnl: float) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "realized_pnl": round(pnl, 2),
            "win": int(pnl > 0),
            "loss": int(pnl < 0),
            "neutral": int(pnl == 0),

        }


    def summary(self, metrics=[]) -> Dict[str, Any]:
        equity_curve = self.portfolio.history
        realized_pnl = self.portfolio.realized_pnl

        if not equity_curve:
            return {}

        # metrics = metrics or self.DEFAULT_METRICS
        # results = []

        # equity_curve_df = pd.DataFrame(realized_pnl, columns=['datetime', 'equity'])
        # print(equity_curve_df)
        #
        # for symbol, group in equity_curve_df.groupby("symbol"):
        #     values = group["price"]
        #     returns = self.calculate_returns(values)
        #
        #     start_day = group["date"].head(1).values[0]
        #     end_day = group["date"].tail(1).values[0]
        #     row = {"symbol": symbol,
        #            "start_day": start_day,
        #            "end_date": end_day,
        #            "period": end_day - start_day,
        #            "start_value": values.head(1).values[0],
        #            "end_value": values.tail(1).values[0]
        #            }
        #
        #     for m in metrics:
        #         if not hasattr(self, m):
        #             raise ValueError(f"Metric '{m}' not found on PerformanceAnalyzer")
        #         func = getattr(self, m)
        #         # choose argument based on signature
        #         if m == "max_drawdown":
        #             row[m] = func(values)
        #         else:
        #             row[m] = func(returns)
        #
        #     results.append(row)
        # print(results)

        initial_cash = equity_curve[0][1]
        final_equity = equity_curve[-1][1]
        total_return = (final_equity - initial_cash) / initial_cash

        # Analyze trade performance
        trade_profits = list(realized_pnl.values())
        wins = [p for p in trade_profits if p > 0]
        losses = [abs(p) for p in trade_profits if p < 0]
        neutral = [p for p in trade_profits if p == 0]

        num_trades = len(trade_profits)
        win_rate = len(wins) / num_trades if num_trades else 0.0
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        profit_factor = (sum(wins) / sum(losses)) if losses else float("inf")

        # Per-symbol breakdown
        symbol_stats = {
            sym: self._analyze_symbol(sym, pnl)
            for sym, pnl in realized_pnl.items()
        }

        results = {
            "initial_cash": round(initial_cash, 2),
            "final_equity": round(final_equity, 2),
            "total_return": round(total_return, 4),
            "num_trades": num_trades,
            "win_rate": round(win_rate, 4),
            "average_win": round(avg_win, 2),
            "average_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 4),
            "per_symbol": symbol_stats
        }
        print(results)

        return  results

    def export_metrics(self, filepath: str = '') -> None:
        import json
        summary = self.summary()

        if not filepath:
            filepath = "stats/performance.json"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        export_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **summary,
        }

        with open(filepath, "w") as f:
            json.dump(export_record, f, indent=2)

    def notify(self):
        summary = self.summary()
        print("🔔 Performance Summary:")
        print(f"Total Return: {summary['total_return'] * 100:.2f}%")
        print(f"Max Drawdown: {summary['max_drawdown'] * 100:.2f}%")
        print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        print("Per-symbol Stats:")
        for symbol, stats in summary["per_symbol"].items():
            print(f"  {symbol}: {stats}")

    #     # Upload to S3
    #     if self.config.aws.s3_upload_enabled:
    #         self._upload_to_s3(versioned_path, f"reports/{filename}")
    #         self._upload_to_s3(latest_path, "reports/performance_latest.json")
    #
    # def _upload_to_s3(self, local_path: str, s3_key: str):
    #     aws = self.config.aws
    #     try:
    #         s3 = boto3.client(
    #             "s3",
    #             aws_access_key_id=aws.access_key,
    #             aws_secret_access_key=aws.secret_key,
    #             region_name=aws.region,
    #         )
    #         s3.upload_file(local_path, aws.s3_bucket, s3_key)
    #         print(f"✅ Uploaded {local_path} to s3://{aws.s3_bucket}/{s3_key}")
    #     except ClientError as e:
    #         print(f"❌ Failed to upload to S3: {e}")

    # def summary(self, metrics: List[str] = None) -> pd.DataFrame:
    #     """
    #     Compute all requested metrics for each symbol and return a DataFrame.
    #
    #     :param metrics: list of metric names; must match one of the public
    #                     methods on this class.  If None, uses DEFAULT_METRICS.
    #     """
    #     metrics = metrics or self.DEFAULT_METRICS
    #     results = []
    #     for symbol, group in self.portfolio.equity_curve.groupby("symbol"):
    #         values = group["price"]
    #         returns = self.calculate_returns(values)
    #
    #         start_day = group["date"].head(1).values[0]
    #         end_day = group["date"].tail(1).values[0]
    #         row = {"symbol": symbol,
    #                "start_day": start_day,
    #                "end_date": end_day,
    #                "period": end_day - start_day,
    #                "start_value": values.head(1).values[0],
    #                "end_value": values.tail(1).values[0]
    #
    #                }
    #         for m in metrics:
    #             if not hasattr(self, m):
    #                 raise ValueError(f"Metric '{m}' not found on PerformanceAnalyzer")
    #             func = getattr(self, m)
    #             # choose argument based on signature
    #             if m == "max_drawdown":
    #                 row[m] = func(values)
    #             else:
    #                 row[m] = func(returns)
    #
    #         results.append(row)
    #
    #     # Convert results into a Pandas DataFrame
    #     return pd.DataFrame(results).set_index("symbol")

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

    def calculate_return(self, dataframe):
        return dataframe.pct_change().fillna(0)

    def calcuate_cumulative_return(self, returns):
        return (1 + returns).cumprod()

    def calculate_drawdown(self, cumulative):
        return cumulative / cumulative.cummax() - 1

    def calculate_total_return(self, cumulative):
        return cumulative.iloc[-1] - 1

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.01) -> float:
        excess = returns - risk_free_rate / self.days
        return np.sqrt(self.days) * excess.mean() / excess.std()

    def calculate_annual_volatility(self, returns):
        return returns.std() * np.sqrt(self.days)

    def calculate_annual_return(self, returns):
        return returns.mean() * self.days
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
    # def plot(self) -> None:
    #     fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    #
    #     self.df["cumulative"].plot(ax=ax[0], title="Cumulative Returns", color="blue")
    #     ax[0].set_ylabel("Cumulative Return")
    #
    #     self.df["drawdown"].plot(ax=ax[1], title="Drawdown", color="red")
    #     ax[1].set_ylabel("Drawdown")
    #     ax[1].set_xlabel("Date")
    #
    #     plt.tight_layout()
    #     plt.show()
#
