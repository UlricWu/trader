# trader/strategy.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, List

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from trader.events import AnalyticsEvent, SignalEvent, EventType
from trader.analytics.feature_store import FeatureStore
from utilts.logs import logs


class BaseStrategy(ABC):
    """
    Abstract base class for all strategies.
    Strategies no longer store their own rolling data,
    they query FeatureStore for feature history.
    """

    def __init__(self, settings, feature_store: FeatureStore, train_window: int = 30, name: str = "Base"):
        self.settings = settings
        self.feature_store = feature_store
        self.train_window = train_window
        # Each symbol may have its own estimator if ML
        self.estimators: Dict[str, object] = {}

        self.name = name

    def on_analytics(self, event: AnalyticsEvent) -> Optional[SignalEvent]:
        """
        Subscribed to AnalyticsEvent (already includes adjusted OHLCV & engineered features).
        """
        if event.type != EventType.ANALYTICS:
            return None

        # delegate to concrete strategy
        return self._generate_signal(event)

    @abstractmethod
    def _generate_signal(self, event: AnalyticsEvent) -> Optional[SignalEvent]:
        raise NotImplementedError


class MLStrategy(BaseStrategy):
    def __init__(self, settings, feature_store: FeatureStore, train_window: int = 60, feature_cols: List[str] = None, name="MLStrategy"):
        super().__init__(settings, feature_store, train_window, name)
        self.feature_cols = feature_cols or ["ma_5", "ma_10", "return_1d"]
        self.train_window = train_window
        self.min_confidence = 0.5

    def _get_model(self, symbol: str):
        if symbol not in self.estimators:
            self.estimators[symbol] = RandomForestClassifier(n_estimators=100, random_state=42)
        return self.estimators[symbol]

    def _generate_signal(self, event: AnalyticsEvent) -> Optional[SignalEvent]:
        symbol = event.symbol
        # Fetch enough history from feature_store (point-in-time)
        hist = self.feature_store.history(symbol, window=self.train_window + 1, end=event.datetime)

        if hist.empty or len(hist) < self.train_window:
            return None

        # build estimator if not exist
        if symbol not in self.estimators:
            self.estimators[symbol] = RandomForestClassifier(n_estimators=100, random_state=42)

        df = hist.copy()
        # build a simple next-bar target if not present
        if "target" not in df.columns:
            df["target"] = (df["close"].pct_change().shift(-1) > 0).astype(int)

        X = df[self.feature_cols]
        y = df["target"].astype(int)
        if y.nunique() < 2 or len(X) < self.train_window:
            return None

        # train on everything up to last-1 row
        X_train, y_train = X.iloc[:-1], y.iloc[:-1]
        X_test = X.iloc[[-1]]

        model = self._get_model(symbol)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[0][1] if hasattr(model, "predict_proba") else None

        if prob is None:
            return None

        conf = max(prob, 1 - prob)
        if conf < self.min_confidence:
            logs.record_log(f"[MLStrategy] skip {symbol} at {event.datetime} prob={prob:.3f} conf={conf:.3f}", 2)
            return None

        side = "BUY" if prob > 0.5 else "SELL"
        return SignalEvent(symbol=symbol, datetime=event.datetime, signal_type=side, source=self.name)


#         #     def summary(self):
#         #         correct = sum(1 for p, a in self.predictions if p == a)
#         #         accuracy = correct / len()
#         #         avg_conf = sum(abs(prob - 0.5) for prob, _ in self.predictions) / len(self.predictions) * 2
#         #
#         #         print(f"ML Prediction Accuracy: {accuracy:.2%} ({correct}/{len(self.predictions)})")
#         #         print(f"Avg Prediction Confidence: {avg_conf:.2%}")
#


class RuleStrategy(BaseStrategy):
    def __init__(self, settings, feature_store: FeatureStore, short_window: int = 5, long_window: int = 20,name='rule-base'):
        super().__init__(settings, feature_store, train_window=long_window, name=name)
        self.short_window = short_window
        self.long_window = long_window

    def _generate_signal(self, event: AnalyticsEvent) -> Optional[SignalEvent]:
        symbol = event.symbol
        # get last `long` rows up to this timestamp (point-in-time)
        df = self.feature_store.history(symbol, window=self.long_window, end=event.datetime)
        if df is None or len(df) < self.long_window:
            return None

        short_ma = df["close"].rolling(self.short_window).mean().iloc[-1]
        long_ma = df["close"].rolling(self.long_window).mean().iloc[-1]

        if pd.isna(short_ma) or pd.isna(long_ma):
            return None

        signal_type = "HOLDING"
        if short_ma > long_ma:
            signal_type = "BUY"  # limit_price = event.close * (1 + self.slippage)  # Buy 1% above the close price
        elif short_ma < long_ma:
            signal_type = "SELL"

        return SignalEvent(symbol=symbol, datetime=event.datetime, signal_type=signal_type, source=self.name)
