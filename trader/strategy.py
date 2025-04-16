# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @File    : strategy.py
# # @Project : trader
# # @Author  : wsw
# # @Time    : 2025/3/12 14:26
# import pandas as pd
# import numpy as np
#
# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd
#
# from abc import ABC, abstractmethod
# from typing import List
# import pandas as pd
#
# from typing import Protocol, List
# import pandas as pd
# from datetime import datetime
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
#
#
# class Strategy:
#
#     @abstractmethod
#     def sizing(self, df: pd.DataFrame, symbol: str, equity: float) -> int:
#         """Determine position sizing based on equity."""
#         pass
#
#     @staticmethod

#
#     @staticmethod
#     def generate_signals(a, b):
#         """Generate signals based on historical data."""
#         pass
#         # if a is None or b is None:
#         #     return 0
#         # if a > b: return 1
#         #
#         # if a < b: return -1
#
#     @abstractmethod
#     def train(self, df: pd.DataFrame):
#         pass
#
#
# class RuleBasedStrategy(Strategy):
#     def train(self, df: pd.DataFrame) -> None:
#         # Implement rule-based logic, like calculating moving averages
#         pass
#
#     # def predict_signal(self, features: pd.DataFrame, symbol: str) -> int:
#     #     # Implement rule-based signal generation logic (buy, sell, hold)
#     #     pass
#
#     def sizing(self, features: pd.DataFrame, symbol: str, equity: float) -> int:
#         # Define position sizing for rule-based strategy
#         pass
#
#
# from sklearn.linear_model import LogisticRegression
#
#
# class MLStrategy(Strategy):
#     def __init__(self):
#         self.models = {}
#         self.features = ['return', 'moving_avg', 'volatility']
#
#     def train(self, df: pd.DataFrame) -> None:
#         for symbol in df["ts_code"].unique():
#             model = RandomForestClassifier(n_estimators=100, random_state=42)
#             data = df[df["ts_code"] == symbol].copy()
#
#             data = self.generate_features(data)
#             #
#             X = data[self.features]
#             y = (data['return'] > 0).astype(int)  # 1 for up, 0 for down
#
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             model.fit(X_train, y_train)
#             # y_pred = model.predict(X_test)
#             # print(f"Model accuracy: {accuracy_score(y_test, y_pred)}")
#
#             self.models[symbol] = model
#
#     def generate_features(self, df) -> pd.DataFrame:
#         """
#         Generate features for the ML model.
#         """
#         df['return'] = df['close'].pct_change()
#         df['moving_avg'] = df['close'].rolling(window=20).mean()
#         df['volatility'] = df['return'].rolling(window=20).std()
#         return df.dropna()
#
#     def predict_signal(self, df: pd.DataFrame, symbol: str) -> int:
#         """
#          Generate trading signals (1 for buy, 0 for hold, -1 for sell).
#          """
#         model = self.models.get(symbol)
#         if model is None:
#             return 0
#
#         features = self.generate_features(df)
#         # X = features[["close"]].pct_change().fillna(0).values[-1].reshape(1, -1)
#         prediction = model.predict(features)[0]
#         return 1 if prediction == 1 else -1
#
#     def sizing(self, features: pd.DataFrame, symbol: str, equity: float) -> int:
#         price = features.iloc[-1]["close"]
#         return int((equity * 0.01) // price) if price > 0 else 0
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
# #
# #     def sizing(self, df: pd.DataFrame, symbol: str, equity: float) -> int:
# #         """Calculate position size based on equity and current price."""
# #         return self.position_sizer.calculate(df, symbol, equity)
#
# # class MLStrategy(Strategy):
# #     def __init__(self):
# #         self.model = RandomForestClassifier(n_estimators=100, random_state=42)
# #
# #         super.__init__()
# #
# #     def train(self, features: pd.DataFrame):
# #         features = features.copy()
# #         features["target"] = (features["return_1d"].shift(-1) > 0).astype(int)
# #         features = features.dropna()
# #         X = features[["sma_ratio"]]
# #         y = features["target"]
# #         self.model.fit(X, y)
# #
# #     def predict_signal(self, features: pd.DataFrame) -> int:
# #         """
# #         Predicts whether to buy (1), sell (-1), or hold (0).
# #         """
# #         latest = features.tail(1)[["sma_ratio"]]
# #         pred = self.model.predict_proba(latest)[0][1]  # Probability stock goes up
# #         return self.generate_signal_threshold(pred)
