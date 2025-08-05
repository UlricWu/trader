# strategy.py
import os
import pickle
from abc import ABC, abstractmethod
import re

import joblib

from trader.events import SignalEvent, EventType
from collections import defaultdict
from utilts.logs import logs
from trader.config import Settings
from trader.model import Model
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class BaseStrategy(ABC):
    def __init__(self, events=None, settings: Settings = None):
        self.events = events
        self.settings = settings

    def put_event(self, event):
        """Safely put event into the queue if events queue exists."""
        if self.events:
            self.events.put(event)

    @abstractmethod
    def on_market(self, event):
        pass


class MLSignalGenerator:
    def __init__(self, train_window: int = 60):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.train_window = train_window  # How many past bars to use for training
        self.features = ["MA5", "MA10", "Return_1d"]
        self.trained = False

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Return_1d"] = df["close"].pct_change()
        df["MA5"] = df["close"].rolling(5).mean()
        df["MA10"] = df["close"].rolling(10).mean()
        df["Target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        return df.dropna()

    def train_and_predict(self, df: pd.DataFrame) -> int:
        df = self.prepare_features(df)
        if len(df) < self.train_window + 1:
            return 0  # Not enough data to train

        # Split: train on [:-1], test on [-1] row
        train_df = df.iloc[-(self.train_window + 1):-1]
        test_df = df.iloc[-1:]

        X_train = train_df[self.features]
        y_train = train_df["Target"]
        X_test = test_df[self.features]

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)[0]
        self.trained = True
        return pred  # 1: Buy, 0: Hold/Sell

    def train_and_predict_proba(self, df: pd.DataFrame) -> float:
        df = self.prepare_features(df)
        if len(df) < self.train_window + 1:
            return 0.5  # Neutral probability if insufficient data

        train_df = df.iloc[-(self.train_window + 1):-1]
        test_df = df.iloc[-1:]

        X_train = train_df[self.features]
        y_train = train_df["Target"]
        X_test = test_df[self.features]

        self.model.fit(X_train, y_train)
        prob_up = self.model.predict_proba(X_test)[0][1]  # P(price up)
        return prob_up


class MLStrategy(BaseStrategy):
    def __init__(self, events, settings: Settings, model=MLSignalGenerator):
        super().__init__(events, settings)
        self.prices = defaultdict(list)
        self.window = settings.model.windows
        self.signal_generator = model()
        self.predictions = []  # (predicted, actual)

        self.current_position = "FLAT"  # or "LONG"
        self.buy_threshold = self.settings.model.buy_threshold
        self.sell_threshold = self.settings.model.sell_threshold

    def on_market(self, event):
        if event.type != EventType.MARKET:
            logs.record_log(f"Skipping {event}", 3)
            return

        # Update the price history for each symbol
        self.prices[event.symbol].append(event)

        if len(self.prices[event.symbol]) <= self.signal_generator.train_window + 1:
            return

        # 1. Feature generation
        df = self._bars_to_dataframe(self.prices[event.symbol])

        # 2. Prediction
        pred = self.signal_generator.train_and_predict(df)
        actual = df["Target"].iloc[-1]  # ground truth from last bar
        self.predictions.append((pred, actual))

        # 3. Send Signal
        if pred == 1 and self.current_position == "FLAT":
            # Generate LONG signal
            signal = SignalEvent(symbol=event.symbol, datetime=event.datetime, signal_type="BUY")
            self.events.put(signal)
            self.current_position = "LONG"
        elif pred == 0 and self.current_position == "LONG":
            # Generate EXIT signal
            signal = SignalEvent(symbol=event.symbol, datetime=event.datetime, signal_type="SELL")
            self.events.put(signal)
            self.current_position = "FLAT"
        # Else, hold position

    def on_market_prediction(self, event):
        if event.type != EventType.MARKET:
            logs.record_log(f"Skipping {event}", 3)
            return

        # Update the price history for each symbol
        self.prices[event.symbol].append(event)

        if len(self.prices[event.symbol]) <= self.signal_generator.train_window + 1:
            return
        df = self._bars_to_dataframe(self.prices[event.symbol])
        pred = self.signal_generator.train_and_predict_proba(df)
        actual = df["Target"].iloc[-1]  # ground truth from last bar
        self.predictions.append((pred, actual))

        if pred >= self.buy_threshold and self.current_position == "FLAT":
            # Generate LONG signal
            signal = SignalEvent(symbol=event.symbol, datetime=event.datetime, signal_type="BUY")
            self.events.put(signal)
            self.current_position = "LONG"
        elif pred == 0 and self.current_position == "LONG":
            # Generate EXIT signal
            signal = SignalEvent(symbol=event.symbol, datetime=event.datetime, signal_type="SELL")
            self.events.put(signal)
            self.current_position = "FLAT"

    def _bars_to_dataframe(self, lists):
        return pd.DataFrame(lists)[["open", "high", "low", "close"]]


class Strategy(BaseStrategy):
    slippage = 0.01

    def __init__(self, events, settings: Settings):
        super().__init__(events, settings)
        self.prices = defaultdict(list)
        self.window = settings.trading.WINDOWS

    def on_market(self, event):
        if event.type != EventType.MARKET:
            logs.record_log(f"Skipping {event}", 3)
            return

        # Update the price history for each symbol
        self.prices[event.symbol].append(event.close)

        if len(self.prices[event.symbol]) < self.window:
            # a guard clause that ensures enough data exists before making a decision.
            logs.record_log(f"Skipping {event} because there are price {self.prices} less than {self.window}", 2)
            return

        avg = sum(self.prices[event.symbol][-self.window:]) / self.window
        if event.close > avg:
            # limit_price = event.close * (1 + self.slippage)  # Buy 1% above the close price
            self.events.put(SignalEvent(symbol=event.symbol, datetime=event.datetime, signal_type="BUY"))
        elif event.close < avg:
            # limit_price = event.close * (1 - self.slippage)  # Sell 1% below the close price
            self.events.put(SignalEvent(symbol=event.symbol, datetime=event.datetime, signal_type="SELL"))

#
# class MLStrategy(BaseStrategy):
#     def __init__(self, events, settings, symbol: str):
#         super().__init__(events)
#         self.symbol = symbol
#         self.settings = settings
#         self.model = Model(settings, symbol)
#
#     def generate_signals(self, market_event):
#         features = self.extract_features(market_event)
#         prediction = self.model.predict([features])[0]
#
#         if prediction == 1:
#             self.create_buy_signal(market_event.symbol)
#         elif prediction == -1:
#             self.create_sell_signal(market_event.symbol)
#
#     def extract_features(self, market_event):
#         return [
#             market_event.open,
#             market_event.high,
#             market_event.low,
#             market_event.close,
#             market_event.volume,
#         ]
#
#     def create_buy_signal(self, symbol):
#         # Emit SignalEvent...
#         ...
#
#     def create_sell_signal(self, symbol):
#         # Emit SignalEvent...
#         ...
