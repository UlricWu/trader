#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : signal_generator.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/17 22:57
# !filepath trader/signal_generator.py
from __future__ import annotations
import abc

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from trader.config import Settings
from utilts import logs


class SignalGenerator(abc.ABC):
    """
    Base class for all signal generators. Each generator produces
    a DataFrame of signals aligned with market data.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    @abc.abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class RuleBasedSignalGenerator(SignalGenerator):
    """
    SMA crossover strategy using settings for short/long windows.
    """

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.short_window = settings.strategy.short_window
        self.long_window = settings.strategy.long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

        # avg = sum(data[-self.window:]) / self.window
        # lst = data.last()
        # if lst > avg:
        #     # limit_price = event.close * (1 + self.slippage)  # Buy 1% above the close price
        #     self.events.put(SignalEvent(symbol=event.symbol, datetime=event.datetime, signal_type="BUY"))
        # elif event.close < avg:
        #     # limit_price = event.close * (1 - self.slippage)  # Sell 1% below the close price
        #     self.events.put(SignalEvent(symbol=event.symbol, datetime=event.datetime, signal_type="SELL"))


class MLSignalGenerator(SignalGenerator):
    """
    ML-based signal generator using settings for model and training parameters.
    """

    def __init__(self, settings: Settings):
        super().__init__(settings)
        # self.model: BaseEstimator = settings.model.get_model() or RandomForestClassifier(
        #     n_estimators=settings.model.n_estimators,
        #     random_state=settings.model.random_state
        # )
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.train_window = settings.strategy.train_window
        self.features = settings.model.features or ["MA5", "MA10", "Return_1d"]
        self.min_confidence = settings.model.min_confidence_to_trade
        self.trained = False

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["MA5"] = df["close"].rolling(5).mean()
        df["MA10"] = df["close"].rolling(10).mean()
        df["Return_1d"] = df["close"].pct_change()
        df["target"] = (df["Return_1d"].shift(-1) > 0).astype(int)
        return df.dropna()

    def _split(self, df):
        # Split: train on [:-1], test on [-1] row
        train_df = df.iloc[-(self.train_window + 1):-1]
        test_df = df.iloc[-1:]
        X_train = train_df[self.features]
        # print(train_df.head())
        y_train = train_df["target"]
        X_test = test_df[self.features]
        return X_test, X_train, y_train

    def generate_signals(self, data: pd.DataFrame) -> float:
        df = self._feature_engineering(data)

        if len(df) < self.train_window + 1:
            return 0  # Not enough data to train
        X_test, X_train, y_train = self._split(df)
        self.model.fit(X_train, y_train)
        prob_up = self.model.predict_proba(X_test)[0][1]  # P(price up)
        return self.confidence(prob_up)

    def confidence(self, pred):

        if pred is None or np.isnan(pred):
            return 0

        # optional min confidence check
        confidence = abs(pred)
        if 0 < confidence <= self.settings.model.min_confidence_to_trade:
            logs.record_log(f'skip because pred probability ={pred} with low confidence={confidence}')
            return 0

        return 1 if pred > 0 else -1
