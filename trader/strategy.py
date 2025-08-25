# # strategy.py
# import os
# import pickle
# from abc import ABC, abstractmethod
# import re
#
# import joblib
# import numpy as np


# ml_strategy.py
from collections import defaultdict
import numpy as np
import pandas as pd

from trader.events import EventType, MarketEvent, SignalEvent, Event
from utilts import logs
from trader.config import Settings

# strategy.py
import os
import pickle
from abc import ABC, abstractmethod
import re

import joblib
import numpy as np

from trader.events import SignalEvent, EventType, Event
from collections import defaultdict
from trader.config import Settings
from trader.model import Model
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from trader.events import EventType, MarketEvent, SignalEvent
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List

from trader.events import EventType, MarketEvent, SignalEvent
from utilts.logs import logs

# trader/analytics/base_strategy.py
from abc import ABC, abstractmethod
from trader.events import FeatureEvent, SignalEvent, EventType
from typing import Deque, Dict, List, Tuple, Optional, Callable
from collections import deque

# trader/analytics/rule_strategy.py
import pandas as pd
# from trader.analytics.base_strategy import BaseStrategy
from trader.events import FeatureEvent, SignalEvent
# trader/analytics/ml_strategy.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from trader.analytics.base_strategy import BaseStrategy
from trader.events import FeatureEvent, SignalEvent


class BaseStrategy(ABC):
    """Abstract base class for all strategies."""

    def __init__(self, settings, train_window=30):
        self.settings = settings
        # Keep rolling buffer of features
        self.train_window = train_window
        self.feature_buffer = deque(maxlen=train_window)
        self.rows: Dict[str, Deque[dict]] = {}
        self.predictions: Dict[str, list[tuple]] = {}
        self.estimator = {}

    def on_market(self, event: FeatureEvent) -> None:
        """Only handle FEATURE events, then delegate to concrete logic."""
        if event.type != EventType.FEATURE or event is None:
            print(f'skip event: {event}')
            return
        self.handle_features(event)

        return self._generate_signal(event)

    @abstractmethod
    def _generate_signal(self, event: FeatureEvent) -> None:
        """Implemented by concrete strategy classes."""
        raise NotImplementedError

    # @abstractmethod
    def handle_features(self, event: FeatureEvent) -> None:

        s = event.symbol
        if s not in self.rows:
            self.rows[s] = deque(maxlen=self.train_window)
            # self.counter[s] = 0
            # self.[s] = self.model_builder()

        # Append one row (features dict must include a "Close" for target construction)
        self.rows[s].append(dict(event.features))

    def get_features(self, symbol: str) -> pd.DataFrame:
        return pd.DataFrame(self.rows[symbol])


class MLStrategy(BaseStrategy):
    def __init__(self, settings, train_window: int = 60):
        super().__init__(settings)
        # self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.train_window = train_window
        self.features = ["MA5", "MA10", "Return_1d"]

    def _generate_signal(self, event: FeatureEvent) -> None:

        # if not self.trained:
        symbol = event.symbol

        df = self.get_features(symbol)

        if len(df) < self.train_window:
            return

        if symbol not in self.estimator:
            self.estimator[symbol] = RandomForestClassifier(n_estimators=100, random_state=42)

        X_test, X_train, y_train, y_test = self._split(df)

        model = self.estimator[symbol]
        model.fit(X_train, y_train)
        self.estimator[symbol] = model

        prob_up = model.predict_proba(X_test)[0][1]  # P(price up)
        # optional min confidence check
        confidence = abs(prob_up)
        if 0 < confidence <= self.settings.model.min_confidence_to_trade:
            logs.record_log(f'skip event={event} because pred probability ={prob_up} with low confidence={confidence}')
            return

        signal = "BUY" if prob_up > 0 else "SELL"
        return SignalEvent(symbol=event.symbol, datetime=event.datetime, signal_type=signal)

        #     #     # last_signal = signals["signal"].iloc[-1]
        #         if pred == 1 and self.current_position == "FLAT":
        #             self.current_position = "LONG"
        #             signal = "BUY"
        #         elif pred == 0 and self.current_position == "LONG":
        #             self.current_position = "FLAT"
        #     def summary(self):
        #         correct = sum(1 for p, a in self.predictions if p == a)
        #         accuracy = correct / len()
        #         avg_conf = sum(abs(prob - 0.5) for prob, _ in self.predictions) / len(self.predictions) * 2
        #
        #         print(f"ML Prediction Accuracy: {accuracy:.2%} ({correct}/{len(self.predictions)})")
        #         print(f"Avg Prediction Confidence: {avg_conf:.2%}")

    def _split(self, df):
        # Split: train on [:-1], test on [-1] row
        df['Target'] = (df["Return_1d"].shift(-1) > 0).astype(int)
        train_df = df.iloc[-(self.train_window + 1):-1]

        test_df = df.iloc[-1:]
        X_train = train_df[self.features]
        y_train = train_df["Target"]
        X_test = test_df[self.features]
        y_test = test_df["Target"]
        return X_test, X_train, y_train, y_test


class RuleStrategy(BaseStrategy):
    def __init__(self, settings, short_window: int = 5, long_window: int = 20):
        super().__init__(settings)
        self.window = settings.trading.WINDOWS

    def _generate_signal(self, event: FeatureEvent) -> None:
        # skip_event = Event(None, None)  # hold
        s = event.symbol
        df = self.get_features(s)

        if len(df) < self.window:
            return

        avg = sum(df['close'][-self.window:]) / self.window
        signal_type = 'HOLDING'

        if event.features['close'] > avg:
            signal_type = "BUY"
            # limit_price = event.close * (1 + self.slippage)  # Buy 1% above the close price
            # self.events.put(SignalEvent(symbol=event.symbol, datetime=event.datetime, signal_type="BUY"))
        elif event.features['close'] < avg:
            signal_type = "SELL"
        return SignalEvent(symbol=event.symbol, datetime=event.datetime, signal_type=signal_type)
