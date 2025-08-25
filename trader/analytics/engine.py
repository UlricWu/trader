#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : engine.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/25 10:59
#!filepath: trader/analytics/engine.py
from collections import defaultdict, deque
from dataclasses import dataclass
import pandas as pd
from typing import Dict, Deque, Optional

# from trader.event_bus import EventBus
from trader.events import MarketEvent, AnalyticsEvent
from trader.analytics.feature_store import FeatureStore
from trader.analytics.adjustment import Adjuster, AdjustmentPolicy

# @dataclass
class AnalyticsEngine:
    """
    AnalyticsEngine computes adjusted price and features per MarketEvent,
    writes them into FeatureStore and publishes AnalyticsEvent via EventBus.

    - bus: EventBus instance (must support subscribe/publish)
    - feature_store: shared FeatureStore instance (injected)
    - train_window controls how much history the engine retains for feature rolling.
    - pipeline_id: version label for features (useful for auditing).
    """
    # bus: EventBus
    feature_store: FeatureStore
    train_window: int = 240
    adjuster: Adjuster = Adjuster(AdjustmentPolicy.HFQ)

    def __init__(self, feature_store: FeatureStore):
        self.buffers: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=self.train_window))
        self.factor_series: Dict[str, pd.Series] = {}  # symbol -> F(t)
        self.feature_store = feature_store
        # self.bus.subscribe(MarketEvent, self.on_market)

    def register_factor_series(self, symbol: str, factors: pd.Series):
        """Register cumulative adjustment factor series for a symbol (index=timestamp)."""
        # factors: 累计因子，index=Timestamp，已按日期对齐
        self.factor_series[symbol] = factors.sort_index()

    def on_market(self, event: MarketEvent):
        sym, ts = event.symbol, pd.Timestamp(event.datetime)
        raw_close = event.close

        # F = self.factor_series.get(sym)
        # if F is not None and ts in F.index:
        #     window_start = (ts if not self.buffers[sym] else self.buffers[sym][0]["datetime"])
        #     close_adj = self.adjuster.adjust_close(
        #         date=ts, raw_close=raw_close, factor_series=F,
        #         window_start=pd.Timestamp(window_start), now=ts
        #     )
        # else:
        #     close_adj = raw_close # todo adjustment

        close_adj = raw_close

        row = {
            "datetime": ts,
            "symbol": sym,
            "close": float(raw_close),
            "close_adj": float(close_adj),
            # "volume": float(event.volume), # include volume/other raw fields if needed
        }
        self.buffers[sym].append(row)

        df = pd.DataFrame(self.buffers[sym]).set_index("datetime").sort_index()

        # === 基本特征（可继续扩展） ===
        df["ret_1"]  = df["close_adj"].pct_change()
        df["ma_5"]   = df["close_adj"].rolling(5).mean()
        df["ma_10"]  = df["close_adj"].rolling(10).mean()
        df["vol_20"] = df["ret_1"].rolling(20).std()
        df["return_1d"] = df["close_adj"].pct_change().iloc[-1]
        # "Volatility" in name:
        #             features[name] = df["price"].pct_change().rolling(window).std().iloc[-1]

        latest = df.iloc[-1]
        features = {
            "close": float(latest["close"]),
            "close_adj": float(latest["close_adj"]),
            "return_1d":float(latest["return_1d"]),
            "ret_1": float(latest["ret_1"]) if pd.notna(latest["ret_1"]) else 0.0,
            "ma_5": float(latest["ma_5"]) if pd.notna(latest["ma_5"]) else 0.0,
            "ma_10": float(latest["ma_10"]) if pd.notna(latest["ma_10"]) else 0.0,
            "vol_20": float(latest["vol_20"]) if pd.notna(latest["vol_20"]) else 0.0,
        }

        # === 目标（监督学习标签：next-bar up）===
        # optionally compute supervised label (next bar up) if you want stored label
        target = None
        # if len(df) >= 2:
        #     next_up = df["close_adj"].iloc[-1] > df["close_adj"].iloc[-2]
        #     target = int(bool(next_up))
        # here we compute label for current row relative to next (helpful for backtest training)
        # But avoid using a label that leaks future info; target is stored as of this timestamp meaning 'future' value available later.

        # 写入 FeatureStore
        self.feature_store.update(sym, timestamp=ts, features=features)

        # 发布 AnalyticsEvent
        return AnalyticsEvent(symbol=sym, datetime=ts, features=features, target=target)
        # self.bus.publish(AnalyticsEvent(symbol=sym, datetime=ts, features=features, target=target))
