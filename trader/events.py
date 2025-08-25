#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : events.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/20 22:49

# events.py
from dataclasses import dataclass
from enum import Enum
import datetime
from typing import Any, Dict, Optional


class EventType(str, Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    SNAPSHOT = 'SNAPSHOT'  # <-- NEW
    FEATURE = "FEATURE"
    ML_FEATURE = "ML_FEATURE"
    ANALYTICS = "ANALYTICS"


@dataclass
class Event:
    type: EventType | None
    datetime: datetime

    def is_empty(self) -> bool:
        """
        Check if the event is 'empty' (not fully initialized).
        Useful for safely handling default or placeholder events.
        """
        return self.type is None or self.datetime is None

    def no_empty(self) -> bool:
        return self.type and self.datetime


@dataclass
class FeatureEvent(Event):
    symbol: str = ""
    features: Dict[str, Any] = None

    def __init__(self, symbol: str, datetime: datetime, features: Dict[str, Any]):
        super().__init__(EventType.FEATURE, datetime)
        self.symbol = symbol
        self.features = features


@dataclass
class MLFeatureEvent(Event):
    symbol: str = ""
    features: Dict[str, Any] = None
    prediction: int = 0
    probability: float = 0.0

    def __init__(
            self,
            symbol: str,
            datetime: datetime,
            features: Dict[str, Any],
            prediction: int,
            probability: float,
    ):
        super().__init__(EventType.ML_FEATURE, datetime, features)
        self.symbol = symbol
        self.features = features
        self.prediction = prediction
        self.probability = probability


@dataclass
class MarketEvent(Event):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    adj_factor: Optional[float] = None  # 可选：当日累计复权因子

    def __init__(self, datetime, symbol, open, high, low, close):
        super().__init__(EventType.MARKET, datetime)
        self.symbol = symbol
        self.open = open
        self.high = high
        self.low = low
        self.close = close


@dataclass
class AnalyticsEvent(Event):
    symbol: str
    datetime: datetime
    features: Dict[str, float]  # e.g. {"close_raw":..., "close_adj":..., "ma_5":...}
    target: Optional[int] = None  # 0/1 标签，或 None（实盘可为空）

    def __init__(self, datetime, symbol, features, target):
        super().__init__(EventType.ANALYTICS, datetime)
        self.symbol = symbol
        self.features = features
        self.target = target


@dataclass
class MLFeatureEvent(Event):
    symbol: str
    datetime: datetime
    prediction: int  # 1 / -1
    probability: float  # 0~1
    meta: Dict[str, Any]

    def __init__(self, symbol, datetime, prediction, probability, meta):
        super().__init__(EventType.ML_FEATURE, datetime)
        self.prediction = prediction
        self.probability = probability
        self.meta = meta
        self.symbol = symbol


@dataclass
class SignalEvent(Event):
    symbol: str
    signal_type: str  # "BUY" or "SELL"
    source: str # source model
    limit_price: float = None

    def __init__(self, symbol, datetime, signal_type, source):
        super().__init__(EventType.SIGNAL, datetime)
        self.symbol = symbol
        self.signal_type = signal_type
        self.source = source


@dataclass
class OrderEvent(Event):
    symbol: str
    order_type: str
    quantity: int
    direction: str
    limit_price: float = None  # Optional

    def __init__(self, symbol, order_type, quantity, direction, datetime, limit_price=None):
        super().__init__(EventType.ORDER, datetime)
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.limit_price = limit_price


@dataclass
class FillEvent(Event):
    symbol: str
    quantity: int
    price: float
    direction: str
    commission: float = 0

    def __init__(self, symbol, price, quantity, direction, datetime):
        super().__init__(EventType.FILL, datetime)
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.direction = direction
