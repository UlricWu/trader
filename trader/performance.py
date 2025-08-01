#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : performance.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/22 10:46
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

from typing import Dict, Optional, List, Callable, Union

# !performance.py
from typing import Callable, Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime

class PerformanceAnalyzer:
    """
    Analyzes portfolio performance after backtest.
    Automatically computes user-defined or built-in metrics at account and symbol level.
    Daily price is the stock's closing value.

    Return is the percentage change between prices.
    """

    def __init__(self, portfolio, metrics: Dict[str, Callable] = None, symbol_metrics: Dict[str, Callable] = None):
        #         Parameters:
        #         - portfolio: A backtested portfolio object with attributes:
        #             - equity_curve: pd.DataFrame with 'date', 'account_value', 'returns'
        #             - positions: Dict[symbol -> position data]

        self.portfolio = portfolio
        self.equity_df = portfolio.equity_df
        self.realized_pnl = portfolio.realized_pnl
        self.metrics = metrics or self._default_metrics()
        self.symbol_metrics = symbol_metrics or self._default_symbol_metrics()

    # =========================
    # Public Summary Interface
    # =========================
    def account_summary(self) -> Dict[str, float]:
        return {name: func() for name, func in self.metrics.items()}

    def symbol_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for symbol in self.realized_pnl.keys():
            summary[symbol] = {name: func(symbol) for name, func in self.symbol_metrics.items()}
        return summary

    def summary(self) -> Dict[str, Union[Dict[str, float], Dict[str, Dict[str, float]]]]:
        return {
            "account": self.account_summary(),
            "per_symbol": self.symbol_summary()
        }

    # ========================
    # Default Metric Definitions
    # ========================

    def _default_metrics(self) -> Dict[str, Callable]:
        return {
            "total_return": self.total_return,
            "max_drawdown": self.compute_max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "final_equity": lambda: self.equity_df["equity"].iloc[-1] if not self.equity_df.empty else 0.0,
            "initial_cash": lambda: self.portfolio.settings.trading.INITIAL_CASH,
            "total_realized_pnl": lambda: round(sum(self.realized_pnl.values()), 2),
            "annualized_return": self.annualized_return
        }

    def _default_symbol_metrics(self) -> Dict[str, Callable]:
        return {
            "realized_pnl": lambda s: round(self.realized_pnl.get(s, 0.0), 2),
            "trade_count": lambda s: len([t for t in self.portfolio.transactions if t.symbol == s]),
            "total_volume": lambda s: sum(t.quantity for t in self.portfolio.transactions if t.symbol == s),
            "max_drawdown": lambda s: self.compute_max_drawdown(s),
            "annualized_return": lambda s: self.annualized_return(s)
        }

    # ========================
    # Unified Metric Functions
    # ========================
    def _compute_total_return(self, df: DataFrame) -> float:
        if df.empty:
            return 0.0
        start_value = df["equity"].iloc[0]
        end_value = df["equity"].iloc[-1]
        return round((end_value - start_value) / start_value, 6)

    def _compute_annualized_return(self, df: DataFrame) -> float:
        total_return = self._compute_total_return(df)
        days = len(df)
        annual_factor = 252 / days
        annualized_return = (1 + total_return) ** annual_factor - 1
        return round(annualized_return, 6)

    def annualized_return(self, symbol: str = None) -> float:
        if symbol is None:
            df = self.portfolio.equity_df
        else:
            df = self.portfolio.symbol_equity_dfs.get(symbol)
        if df is None or df.empty or len(df) < 2:
            return 0.0
        return self._compute_annualized_return(df)

    def total_return(self, symbol: str = None) -> float:
        """Overall % change over the period"""
        if symbol is None:
            df = self.portfolio.equity_df
        else:
            df = self.portfolio.symbol_equity_dfs.get(symbol)
        if df is None or df.empty or len(df) < 2:
            return 0.0

        return self._compute_total_return(df)

    def _compute_max_drawdown(self, df: DataFrame) -> float:

        cummax = df["equity"].cummax()
        drawdown = df["equity"] / cummax - 1.0
        return round(drawdown.min(), 6)

    def compute_max_drawdown(self, symbol: str = None) -> float:
        if symbol is None:
            df = self.portfolio.equity_df
        else:
            df = self.portfolio.symbol_equity_dfs.get(symbol)
        if df is None or df.empty or len(df) < 2:
            return 0.0
        return self._compute_max_drawdown(df)

    def _compute_daily_returns(self, series: pd.Series) -> pd.Series:
        return series.pct_change().dropna() if not series.empty else pd.Series(dtype=float)

    def daily_returns(self, symbol: str = None) -> pd.Series:
        if symbol is None:
            series = self.portfolio.equity_df
        else:
            series = self.portfolio.symbol_equity_dfs.get(symbol)
        if series is None or series.empty or len(series) < 2:
            return 0.0

        return self._compute_daily_returns(series)

    def sharpe_ratio(self, symbol: str = None, risk_free_rate=0.0) -> float:
        daily_returns = self.daily_returns(symbol)
        if daily_returns.empty:
            return 0.0
        excess = daily_returns - risk_free_rate / 252
        return round(excess.mean() / excess.std() * (252 ** 0.5), 4)[0]  # pd.series->float

#     # ================== Extensibility ==================


# #     def export_metrics(self, filepath: str = '') -> None:
# #         import json
# #         summary = self.summary()
# #
# #         if not filepath:
# #             filepath = "stats/performance.json"
# #         os.makedirs(os.path.dirname(filepath), exist_ok=True)
# #         export_record = {
# #             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
# #             **summary,
# #         }
# #
# #         with open(filepath, "w") as f:
# #             json.dump(export_record, f, indent=2)
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
