# strategy.py
import pandas as pd

from trader.events import SignalEvent, EventType
from collections import defaultdict
from utilts.logs import logs


class Strategy(object):
    slippage = 0.01

    def __init__(self, events, window=3):
        self.events = events
        self.prices = defaultdict(list)

        self.window = window

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
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class MLStrategy:
    def __init__(self, events, model: RandomForestClassifier, window_size=5):
        self.events = events
        self.model = model
        self.window_size = window_size
        self.data = {}

    def on_market(self, event):
        symbol = event.symbol
        price = event.data["close"]
        if symbol not in self.data:
            self.data[symbol] = []
        self.data[symbol].append(price)

        if len(self.data[symbol]) >= self.window_size:
            X = np.array(self.data[symbol][-self.window_size:]).reshape(1, -1)
            pred = self.model.predict(X)[0]
            if pred in ("LONG", "SHORT"):
                signal = SignalEvent(symbol, event.time, pred)
                self.events.put(signal)



# train_ml_model.py
from sklearn.ensemble import RandomForestClassifier
# from trader.config.config import ADJUSTMENT_TYPE, PriceAdjustmentType

# ADJUSTMENT_TYPE = PriceAdjustmentType.QFQ  # Ensure QFQ is used


def prepare_features(df: pd.DataFrame, window=5):
    df = df.copy()
    X, y = [], []
    for i in range(window, len(df)):
        features = df["close"].iloc[i - window:i].values
        label = "LONG" if df["close"].iloc[i] > df["close"].iloc[i - 1] else "SHORT"
        X.append(features)
        y.append(label)
    return np.array(X), np.array(y)


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
