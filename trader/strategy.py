# strategy.py
import os
import pickle
from abc import ABC, abstractmethod
import re

import joblib

from mistune.plugins import math

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

        X_test, X_train, y_train = self._split(df)

        self.model.fit(X_train, y_train)
        pred = self.model.predict(X_test)[0]
        self.trained = True
        return pred  # 1: Buy, 0: Hold/Sell

    def _split(self, df):
        # Split: train on [:-1], test on [-1] row
        train_df = df.iloc[-(self.train_window + 1):-1]
        test_df = df.iloc[-1:]
        X_train = train_df[self.features]
        y_train = train_df["Target"]
        X_test = test_df[self.features]
        return X_test, X_train, y_train

    def train_and_predict_proba(self, df: pd.DataFrame) -> float:
        df = self.prepare_features(df)
        if len(df) < self.train_window + 1:
            return 0.5  # Neutral probability if insufficient data

        X_test, X_train, y_train = self._split(df)

        self.model.fit(X_train, y_train)
        prob_up = self.model.predict_proba(X_test)[0][1]  # P(price up)
        return prob_up


class MLStrategy(BaseStrategy):
    def __init__(self, events, settings: Settings, model=MLSignalGenerator):
        super().__init__(events, settings)
        self.prices = defaultdict(list)

        self.signal_generator = model()
        self.predictions = []  # (predicted, actual)

        self.current_position = "FLAT"  # or "LONG"

        self.prob = settings.model.prob

    def on_market(self, event):
        if event.type != EventType.MARKET:
            logs.record_log(f"Skipping {event}", 3)
            return

        # Update the price history for each symbol
        self.prices[event.symbol].append(event)

        windows = self.settings.model.training_windows
        if len(self.prices[event.symbol]) <= windows + 1:
            logs.record_log(
                f"Skipping {event} because there is not enough data ({len(self.prices[event.symbol])} < {windows})",
                2
            )

            return

        # 1. Feature generation
        df = self._bars_to_dataframe(self.prices[event.symbol])

        signal_type = self._generate_signal(df)
        if  signal_type=='HOLDING':
            logs.record_log(f'holding prediction {event}', )
            return  # error

        signal = SignalEvent(symbol=event.symbol, datetime=event.datetime, signal_type=signal_type)
        self.events.put(signal)

    def _generate_signal(self, df):
        signal = 'HOLD'
        if self.prob:
            pred = self.signal_generator.train_and_predict_proba(df)

            if pred is None or math.isnan(pred):

                return ''

            # optional min confidence check
            confidence = abs(pred - 0.5)
            if 0< confidence <= self.settings.model.min_confidence_to_trade:
                logs.record_log(f'skip because pred probability ={pred} with low confidence={confidence}')
                return ''

            # 3. Send Signal
            if pred == 1 and self.current_position == "FLAT":
                # Generate LONG signal

                self.current_position = "LONG"
                signal = 'BUY'
            elif pred == 0 and self.current_position == "LONG":
                # Generate EXIT signal

                self.current_position = "FLAT"
                signal = 'SELL'
            # Else, hold position

        else:
            pred = self.signal_generator.train_and_predict(df)

            if pred not in {1, 0, -1}:
                logs.record_log('skip because pred is {pred}')
                return signal
            if pred == 1 and self.current_position == "FLAT":
                signal = 'BUY'
            elif pred == 0 and self.current_position == "LONG":
                signal = 'SELL'
                self.current_position = "FLAT"

        actual = df["Target"].iloc[-1]  # ground truth from last bar
        self.predictions.append((pred, actual))

        return signal

    def _bars_to_dataframe(self, lists):

        # ===== Feature Extraction =====
        try:
            # features = [getattr(event, feat) for feat in self.feature_list]
            features = pd.DataFrame(lists)[["open", "high", "low", "close"]]
        except AttributeError as e:
            raise ValueError(f"Missing feature in MarketEvent: {e}")

        return features


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
