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


@dataclass
class MarketEvent(Event):
    datetime: datetime.datetime
    symbol: str
    price: float
    type: EventType = field(default=EventType.MARKET, init=False)


    # def __post_init__(self):
    #     self.type = EventType.MARKET


@dataclass
class SignalEvent(Event):
    datetime: datetime.datetime
    symbol: str
    signal_type: str  # "BUY" or "SELL"
    type: EventType = field(default=EventType.MARKET, init=False)

    # def __post_init__(self):
    #     self.type = EventType.SIGNAL


@dataclass
class OrderEvent(Event):
    datetime: datetime.datetime
    symbol: str
    quantity: int
    direction: str  # "BUY" or "SELL"
    type: EventType = field(default=EventType.MARKET, init=False)

    # def __post_init__(self):
    #     self.type = EventType.ORDER


@dataclass
class FillEvent(Event):
    datetime: datetime.datetime
    symbol: str
    quantity: int
    price: float
    direction: str
    type: EventType = field(default=EventType.MARKET, init=False)

    # def __post_init__(self):
    #     self.type = EventType.FILL
