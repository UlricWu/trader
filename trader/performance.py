#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : performance.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/22 10:46
import itertools
import json
import os

import matplotlib.pyplot as plt

from typing import Dict, Optional, List, Callable, Union

# !performance.py
from typing import Callable, Dict, List, Any
import pandas as pd
from datetime import datetime
import trader.statistics as stats
import seaborn as sns


class PerformanceAnalyzer:
    """
    Analyzes portfolio performance after backtest.
    Automatically computes user-defined or built-in metrics at account and symbol level.
    Daily price is the stock's closing value.

    Return is the percentage change between prices.
    """

    def __init__(self, portfolio, bench=None):
        #         Parameters:
        #         - portfolio: A backtested portfolio object with attributes:
        #             - equity_curve: pd.DataFrame with 'date', 'account_value', 'returns'

        self.portfolio = portfolio
        self.equity_df = portfolio.equity_df
        self.bench = bench

    def summary(self, equity=None):
        if equity is None:
            equity = self.portfolio.symbol_equity_df

        results = {'account': self._summary(equity.sum(axis=1))}
        for col in equity.columns:
            e = equity[col]
            results[col] = self._summary(e)

        return results

    def _summary(self, e):
        daily = stats.daily_returns(e)
        max_dd, dd_dur = stats.max_drawdown_and_duration(e)
        sharpe = stats.sharpe_ratio(e)
        return {
            "equity": e,
            "returns": daily,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "max_ddduration": dd_dur,
            "volatility": stats.volatility(e),
            "annual_return": stats.annual_returns(e),
            "total_return": stats.cumulative_returns(e)
        }

    def plot(self):

        summary = self.summary()

        n_symbols = len(summary.keys())
        n_panels = 3  # Equity, Drawdown, table, Monthly Returns
        fig, axes = plt.subplots(n_panels + n_symbols, 1, figsize=(12, 8))

        # 1. Equity curve
        ax = axes[0]
        for sym, s in summary.items():
            ax.plot(s["equity"].index, s["equity"].values, label=sym)

        ax.set_title("Equity Curve")
        ax.set_ylabel("Equity Value")
        ax.grid(True)
        ax.legend()

        ax = axes[1]

        # dd_acc = acount_summary["equity"] / acount_summary["equity"].cummax() - 1
        # ax.fill_between(dd_acc.index, dd_acc.values, label="ACCOUNT DD", color="black")

        for sym, s in summary.items():
            dd = s["equity"] / s["equity"].cummax() - 1
            if sym == 'account':
                continue
            ax.fill_between(dd.index, dd.values * 100, label=sym, alpha=0.3)
        ax.set_title("Drawdowns")
        ax.legend()

        ax = axes[2]
        self._plot_stats_table(ax, summary)

        # 3. Monthly returns: one heatmap per symbol

        for idx, (sym, s) in enumerate(summary.items()):
            ax = axes[n_panels + idx]

            ax.set_title(f"Monthly Returns Heatmap - {sym}")
            self._plot_monthly_returns(ax, s)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_monthly_returns(ax, stats_dict: dict):
        if ax is None:
            ax = plt.gca()

        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        monthly_returns_df = stats_dict["returns"].resample("ME").apply(lambda x: (1 + x).prod() - 1).to_frame("Return")

        monthly_returns_df["Year"] = monthly_returns_df.index.year
        monthly_returns_df["Month"] = monthly_returns_df.index.strftime("%b")

        pivot_table = monthly_returns_df.pivot(index="Year", columns="Month", values="Return")
        pivot_table = pivot_table.reindex(columns=month_order)

        sns.heatmap(pivot_table, annot=True, fmt=".1%", center=0,
                    cmap="RdYlGn", ax=ax, cbar=False)

    @staticmethod
    def _plot_stats_table(ax, stats_dict: dict):
        if ax is None:
            ax = plt.gca()
        table_data = []
        for sym, s in stats_dict.items():
            table_data.append([
                sym,
                # s['cum_returns'] * 100,
                s['annual_return'],
                s['volatility'],
                s['sharpe'],
                s['max_dd'],
                s['max_ddduration']
            ])
        table_df = pd.DataFrame(table_data, columns=[
            "Symbol", "Annual Return", "Volatility",
            "Sharpe", "Max DD", "Max DD Duration"
        ])
        ax.axis("off")
        table = ax.table(cellText=table_df.values,
                         colLabels=table_df.columns,
                         loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.3)

    # def export_metrics(self, filepath: str = '') -> None:
    #     import json
    #     summary = self.summary()
    #
    #     if not filepath:
    #         filepath = "stats/performance.json"
    #     os.makedirs(os.path.dirname(filepath), exist_ok=True)
    #     export_record = {
    #         "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #         **summary,
    #     }
    #
    #     with open(filepath, "w") as f:
    #         json.dump(export_record, f, indent=2)
# #
# #     def notify(self):
# #         summary = self.summary()
# #         print("🔔 Performance Summary:")
# #         print(f"Total Return: {summary['total_return'] * 100:.2f}%")
# #         print(f"Max Drawdown: {summary['max_drawdown'] * 100:.2f}%")
# #         print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
# #         print("Per-symbol Stats:")
# #         for symbol, stats in summary["per_symbol"].items():
# #             print(f"  {symbol}: {stats}")
# #
# #     #     # Upload to S3
# #     #     if self.config.aws.s3_upload_enabled:
# #     #         self._upload_to_s3(versioned_path, f"reports/{filename}")
# #     #         self._upload_to_s3(latest_path, "reports/performance_latest.json")
# #     #
# #     # def _upload_to_s3(self, local_path: str, s3_key: str):
# #     #     aws = self.config.aws
# #     #     try:
# #     #         s3 = boto3.client(
# #     #             "s3",
# #     #             aws_access_key_id=aws.access_key,
# #     #             aws_secret_access_key=aws.secret_key,
# #     #             region_name=aws.region,
# #     #         )
# #     #         s3.upload_file(local_path, aws.s3_bucket, s3_key)
# #     #         print(f"✅ Uploaded {local_path} to s3://{aws.s3_bucket}/{s3_key}")
# #     #     except ClientError as e:
# #     #         print(f"❌ Failed to upload to S3: {e}")
# #
