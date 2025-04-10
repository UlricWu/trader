#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : strategy.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/3/12 14:26
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class Strategy(ABC):

    @abstractmethod
    def sizing(self, df: pd.DataFrame, symbol: str, equity: float) -> int:
        """Determine position sizing based on equity."""
        pass

    @staticmethod
    def predict_signal(self, prob_up):
        """Return the prediction (buy, sell, hold) for a given symbol."""
        if prob_up > 0.6:
            return 1  # Buy
        elif prob_up < 0.4:
            return -1  # Sell
        else:
            return 0  # Do nothing (neutral) Hold

    @staticmethod
    def generate_signals(self, a, b):
        """Generate signals based on historical data."""
        if a is None or b is None:
            return 0
        if a > b: return 1

        if a < b: return -1

#
# class MLStrategy(Strategy):
#     def __init__(self):
#         self.model = RandomForestClassifier(n_estimators=100, random_state=42)
#
#         super.__init__()
#
#     def train(self, features: pd.DataFrame):
#         features = features.copy()
#         features["target"] = (features["return_1d"].shift(-1) > 0).astype(int)
#         features = features.dropna()
#         X = features[["sma_ratio"]]
#         y = features["target"]
#         self.model.fit(X, y)
#
#     def predict_signal(self, features: pd.DataFrame) -> int:
#         """
#         Predicts whether to buy (1), sell (-1), or hold (0).
#         """
#         latest = features.tail(1)[["sma_ratio"]]
#         pred = self.model.predict_proba(latest)[0][1]  # Probability stock goes up
#         return self.generate_signal_threshold(pred)
