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
from typing import Any, Dict


class EventType(str, Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"
    SNAPSHOT = 'SNAPSHOT'  # <-- NEW
    FEATURE = "FEATURE"
    ML_FEATURE = "ML_FEATURE"


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

    def __init__(self, datetime, symbol, open, high, low, close):
        super().__init__(EventType.MARKET, datetime)
        self.symbol = symbol
        self.open = open
        self.high = high
        self.low = low
        self.close = close


@dataclass
class SignalEvent(Event):
    symbol: str
    signal_type: str  # "BUY" or "SELL"
    limit_price: float = None

    def __init__(self, symbol, datetime, signal_type):
        super().__init__(EventType.SIGNAL, datetime)
        self.symbol = symbol
        self.signal_type = signal_type


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
