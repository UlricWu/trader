# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @File    : strategy.py
# # @Project : trader
# # @Author  : wsw
# # @Time    : 2025/3/12 14:26


from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from configs.config import StrategyConfig

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# disable chained assignments
pd.options.mode.chained_assignment = None
class Strategy(ABC):

    @abstractmethod
    def sizing(self, df: pd.DataFrame, symbol: str, equity: float) -> int:
        """Determine position sizing based on equity."""
        pass

    @abstractmethod
    def train(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame):
        """Generate signals based on historical data."""
        pass


#
# # class RuleBasedStrategy(Strategy):
# #     def __init__(self, short_window: int = 50, long_window: int = 200):
# #         self.feature_engineer = FeatureEngineer()
# #         self.signal_generator = RuleBasedSignalGenerator(short_window, long_window)
# #         self.position_sizer = PositionSizer(risk_percentage=0.01)
# #
# #     def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
# #         df = self.feature_engineer.generate_sma_features(df, short_window=self.signal_generator.short_window,
# #                                                          long_window=self.signal_generator.long_window)
# #         df = self.signal_generator.generate(df)
# #         return df
# #
# #     def predict_signal(self, df: pd.DataFrame, symbol: str) -> int:
# #         """Generate prediction signal based on rule-based strategy (e.g., moving average crossover)."""
# #         latest_signal = df.iloc[-1]["signal"]
# #         return latest_signal

# if a is None or b is None:
#     return 0
# if a > b: return 1
#
# if a < b: return -1
# #
# #     def sizing(self, df: pd.DataFrame, symbol: str, equity: float) -> int:
# #         """Calculate position size based on equity and current price."""
# #         return self.position_sizer.calculate(df, symbol, equity)
#     def train(self, df: pd.DataFrame) -> None:
#         # Implement rule-based logic, like calculating moving averages
#         pass
#
class MLStrategy:

    def __init__(self, config: StrategyConfig):
        self.config = config

        self.features = ['return', 'moving_avg', 'volatility']
        self.target = ['return']
        self.signals = {}

        self.models = {}

        self.scalers: Dict[str, StandardScaler] = {}

    def generate_features(self, df) -> pd.DataFrame:
        """
        Generate features for the ML model.
        """

        df['return'] = df['close'].pct_change().fillna(0)
        df['moving_avg'] = df['close'].rolling(window=20).mean().fillna(method="bfill")
        df['volatility'] = df['return'].rolling(window=20).std().fillna(method="bfill")

        df["future_return"] = df["close"].shift(-1) / df["close"] - 1
        df["target"] = df["future_return"].apply(lambda r: 1 if r > 0 else -1)
        # drop the last row of each symbol (NaN return)
        # df.dropna(subset=["target"]).reset_index(drop=True)
        return df

        # df["sma_5"] = df["close"].rolling(5).mean().fillna(method="bfill")
        # df["sma_10"] = df["close"].rolling(10).mean().fillna(method="bfill")
        #
        # df["feature2"] = df["sma_5"] / df["sma_10"] - 1
        # df["feature3"] = df["close"].diff().fillna(0)

    def train(self, df: pd.DataFrame, symbol: str) -> None:
        df = self.generate_features(df)
        X = df[self.features]

        y = df["target"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        self.models[symbol] = model
        self.scalers[symbol] = scaler

    def predict(self, row: pd.DataFrame, symbol: str) -> int:
        X = row[self.features]
        X_scaled = self.scalers[symbol].transform(X)
        return self.models[symbol].predict_proba(X_scaled)[0]

    def prodict_proba(self, row: pd.DataFrame, symbol: str) -> int:
        """Return the prediction (buy, sell, hold) for a given symbol."""
        prob = self.predict(row, symbol)

        # assume classes_ == [-1, 1] or [0,1] — find index of “up”:
        # up_idx = list(self.models[symbol].classes_).index(1)
        # p_up = prob[up_idx]
        # p_down = 1 - p_up
        #
        # if p_up >= self.config.long_threshold:
        #     return 1
        # elif p_down >= self.config.short_threshold:
        #     return -1
        # else:
        #     return 0

        return self.generate_prob(prob[0])

    #
    def generate_prob(self, prob):
        if prob > self.config.long_threshold:
            return 1  # Buy
        if prob < self.config.short_threshold:
            return -1
        return 0  # Do nothing (neutral) Hold or Assign a default “hold” label for the last row with NAN

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["signal"] = 0

        for symbol, group in df.groupby("symbol"):
            signals = [0] * self.config.window_size
            group = group.sort_values("date").reset_index(drop=True)

            for i in range(self.config.window_size, len(group)):
                window_raw = group.iloc[0:i + 1].copy()  # include i-th row
                window_engineered = self.generate_features(window_raw.iloc[:-1])  # drop last

                self.train(window_engineered, symbol)

                predict_engineered = self.generate_features(window_raw)  # use i-th row

                signal = self.prodict_proba(predict_engineered.iloc[[i]], symbol)
                signals.append(signal)

            df.loc[df["symbol"] == symbol, "signal"] = signals

        return df

    #     def sizing(self, features: pd.DataFrame, symbol: str, equity: float) -> int:
    #         price = features.iloc[-1]["close"]
    #         return int((equity * 0.01) // price) if price > 0 else 0
