#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : events.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/20 22:49

# events.py
from dataclasses import dataclass, field
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
    limit_price: float

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

    def __init__(self, symbol, order_type, quantity, direction, datetime,  limit_price=None):
        super().__init__(EventType.ORDER, datetime)
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.limit_price = limit_price

    # def __str__(self):
    #     return f'datetime={self.datetime} symbol={self.symbol} order_type={self} quantity={self.quantity} direction={self.direction} '
    #

@dataclass
class FillEvent(Event):
    symbol: str
    quantity: int
    price: float
    direction: str

    def __init__(self, symbol, price, quantity, direction, datetime):
        super().__init__(EventType.FILL, datetime)
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.direction = direction

    # def __str__(self):
    #     """Returns a string containing only the non-default field values."""
    #     s = ', '.join(f'{field.name}={getattr(self, field.name)!r}'
    #                   for field in dataclasses.fields(self)
    #                   if getattr(self, field.name) != field.default)
    #
    #     return f'{type(self).__name__}({s})'
    #
    # def __str__(self):
    #     return f'datetime={self.datetime} symbol={self.symbol} order_type={self} quantity={self.quantity} direction={self.direction} '
