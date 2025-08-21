#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : analytics.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/21 12:09
# trader/analytics.py
from typing import List, Tuple
from trader.event_bus import EventBus

class AnalyticsRecorder:
    def __init__(self):
        self.events_seen: List[Tuple[str, str]] = []  # (datetime, type)

    def on_any(self, event, bus: EventBus):
        dt = getattr(event, "datetime", None)
        et = getattr(event, "type", None)
        self.events_seen.append((str(dt), str(et)))
        # You can add per-event metrics here (fills, pnl snapshots, etc.)
