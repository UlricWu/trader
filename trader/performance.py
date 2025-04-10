#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : performance.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/8 23:32
#!performance.py

import numpy as np
import matplotlib.pyplot as plt
from typing import List
from datetime import datetime

class PerformanceAnalyzer:
    def __init__(self, portfolio):
        self.equity_curve = portfolio.equity_curve

    def compute_sharpe_ratio(self):
        pass

    def plot_performance(self):
        pass


#
# class PerformanceAnalyzer:
#     def __init__(self, equity_curve: List[float], dates: List[datetime]):
#         self.returns = self._compute_returns(equity_curve)
#         self.dates = dates
#         self.equity_curve = equity_curve
#
#     def _compute_returns(self, curve: List[float]) -> np.ndarray:
#         return np.diff(curve) / curve[:-1]
#
#     def compute_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
#         excess_returns = self.returns - risk_free_rate / 252
#         return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
#
#     def compute_max_drawdown(self) -> float:
#         cumulative = np.array(self.equity_curve)
#         peak = np.maximum.accumulate(cumulative)
#         drawdown = (cumulative - peak) / peak
#         return np.min(drawdown)
#
#     # def compute_cagr(self) -> float:
#     #     total_return = self.equity_curve[-1] / self.equity_curve[0] - 1
#     #     years = max(1,(self.dates[-1] - self.dates[0]).days / 365.25)
#     #     return (1 + total_return) ** (1 / years) - 1
#
#     def plot_equity_curve(self):
#         plt.figure(figsize=(10, 5))
#         plt.plot(self.dates, self.equity_curve, label="Equity Curve")
#         plt.title("Equity Curve")
#         plt.xlabel("Date")
#         plt.ylabel("Portfolio Value")
#         plt.legend()
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
