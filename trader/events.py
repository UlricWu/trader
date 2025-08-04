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


class EventType(str, Enum):
    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


@dataclass
class Event:
    type: EventType
    datetime: datetime


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
