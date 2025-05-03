# strategy.py
import os
import pickle
from abc import ABC, abstractmethod
import re

from trader.events import SignalEvent
from trader.config import Settings
import numpy as np
import joblib

from trader.events import SignalEvent, EventType
from collections import defaultdict
from utilts.logs import logs
from trader.config import Settings
from trader.model import Model


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
        # else:
        #     print(f"Skipping {event} because error")


from trader.events import SignalEvent


class MLStrategy(BaseStrategy):
    def __init__(self, events, settings, symbol: str):
        super().__init__(events)
        self.symbol = symbol
        self.settings = settings
        self.model = Model(settings, symbol)

    def generate_signals(self, market_event):
        features = self.extract_features(market_event)
        prediction = self.model.predict([features])[0]

        if prediction == 1:
            self.create_buy_signal(market_event.symbol)
        elif prediction == -1:
            self.create_sell_signal(market_event.symbol)

    def extract_features(self, market_event):
        return [
            market_event.open,
            market_event.high,
            market_event.low,
            market_event.close,
            market_event.volume,
        ]

    def create_buy_signal(self, symbol):
        # Emit SignalEvent...
        ...

    def create_sell_signal(self, symbol):
        # Emit SignalEvent...
        ...



#
#     def on_market(self, event):
#         features = self._generate_features(event.bars)
#         prediction = self.model.predict(features)[0]
#
#         signal = SignalEvent(
#             symbol=list(event.bars.keys())[0],
#             datetime=event.bars[list(event.bars.keys())[0]].datetime,
#             signal_type='LONG' if prediction > 0 else 'SHORT'
#         )
#         self.put_event(signal)


# class MLStrategy:
#     def __init__(self, events, settings: Settings):
#         super().__init__(events, settings)
#
#     def on_market(self, event):
#         symbol = event.symbol
#         price = event.data["close"]
#         if symbol not in self.data:
#             self.data[symbol] = []
#         self.data[symbol].append(price)
#
#         if len(self.data[symbol]) >= self.window_size:
#             X = np.array(self.data[symbol][-self.window_size:]).reshape(1, -1)
#             pred = self.model.predict(X)[0]
#             if pred in ("LONG", "SHORT"):
#                 signal = SignalEvent(symbol, event.time, pred)
#                 self.events.put(signal)

#
#     def _generate_features(self, bars):
#         """Extracts features for the model from bars"""
#         features = []
#         for symbol in self.symbol_list:
#             df = bars.get(symbol)
#             if df is not None and len(df) >= self.long_window:
#                 # Example simple feature: moving averages
#                 short_ma = df['close'].rolling(window=self.short_window).mean()
#                 long_ma = df['close'].rolling(window=self.long_window).mean()
#                 features.append(short_ma.iloc[-1])
#                 features.append(long_ma.iloc[-1])
#             else:
#                 features.append(0)
#                 features.append(0)
#         return np.array(features).reshape(1, -1)
#
#     def on_market(self, event):
#         bars = event.data  # assume bars are passed as DataFrame-like dict
#
#         # 1. Feature generation
#         X = self._generate_features(bars)
#
#         # 2. Prediction
#         y_pred = self.model.predict(X)[0] if hasattr(self.model, "predict") else 0
#
#         # 3. Send Signal
#         for symbol in self.symbol_list:
#             if y_pred == 1:
#                 signal = SignalEvent(symbol, "LONG")
#                 self.events.put(signal)
#             elif y_pred == -1:
#                 signal = SignalEvent(symbol, "SHORT")
#                 self.events.put(signal)
#


# def prepare_features(df: pd.DataFrame, window=5):
#     df = df.copy()
#     X, y = [], []
#     for i in range(window, len(df)):
#         features = df["close"].iloc[i - window:i].values
#         label = "LONG" if df["close"].iloc[i] > df["close"].iloc[i - 1] else "SHORT"
#         X.append(features)
#         y.append(label)
#     return np.array(X), np.array(y)

# def _generate_features(self, bars):
#         close_prices = np.array([bar.close for bar in bars.values()])
#         ma_short = np.mean(close_prices[-self.settings.strategy.short_window:])
#         ma_long = np.mean(close_prices[-self.settings.strategy.long_window:])
#         return np.array([ma_short, ma_long]).reshape(1, -1)

# def train_model(data_dict: dict[str, pd.DataFrame], window=5):
#     all_X, all_y = [], []
#     for symbol, df in data_dict.items():
#         if ADJUSTMENT_TYPE == PriceAdjustmentType.QFQ and "qfq_close" in df.columns:
#             df["close"] = df["qfq_close"]
#         X, y = prepare_features(df, window)
#         all_X.append(X)
#         all_y.append(y)
#
#     X_all = np.vstack(all_X)
#     y_all = np.concatenate(all_y)
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_all, y_all)
#     return model
