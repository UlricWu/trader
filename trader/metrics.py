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


class PerformanceAnalyzer:
    days = 252
    risk_free_rate = 0.02

    def __init__(self, trades):
        trades['trade_date'] = pd.to_datetime(trades['trade_date'], format="%Y%m%d").dt.date

        self.trades = trades

    @property
    def statistics(self):
        """Calls all stat functions and returns a DataFrame."""
        stat_properties = [
            "start_date",
            "end_date",
            "period",
            "start_value",
            "end_value",
            "avg_return_d",
            "avg_return",
            "log_return",
            "sharpe_ratio"
        ]

        # Execute properties and collect results
        results = [(stat, getattr(self, stat)) for stat in stat_properties]

        # Convert results into a Pandas DataFrame
        return pd.DataFrame(results, columns=["Stat Name", "Stat Value"])

    @property
    def start_date(self):
        return self.trades['trade_date'].head(1).iloc[0]

    @property
    def end_date(self):
        return self.trades['trade_date'].tail(1).iloc[0]

    @property
    def period(self):

        return (self.end_date - self.start_date).days

    @property
    def start_value(self):
        return self.trades['close'].head(1).iloc[0]

    @property
    def end_value(self):
        return self.trades['close'].tail(1).iloc[0]

    @property
    def avg_return_d(self):
        simple_return = 'simple_return'
        if simple_return not in self.trades.columns.values:
            self.calculate_change_pct()

        return self.trades[simple_return].mean()

    def calculate_change_pct(self):
        self.trades['simple_return'] = self.trades['close'].pct_change()

    @property
    def avg_return(self, days=None):
        if not days:
            days = len(self.trades)
        return self.avg_return_d * days

    @property
    def log_return(self, days=None):
        if not days:
            days = self.days
        log_return = np.log(self.trades['close'] / self.trades['close'].shift(1))
        return log_return.mean() * days

    @property
    def sharpe_ratio(self, risk_free_rate=None, days=None):
        if not days:
            days = self.days
        if not risk_free_rate:
            risk_free_rate = self.risk_free_rate
        returns = self.trades['simple_return']
        return (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(days)
