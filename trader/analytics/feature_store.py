#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : feature_store.py.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/25 10:58
# !filepath: trader/analytics/feature_store.py
import os
from collections import defaultdict
import pandas as pd
from typing import Dict, List, Tuple

from trader.events import FeatureEvent

# !filepath: trader/feature_store.py
from typing import Dict, Any, Optional
import pandas as pd
from collections import defaultdict
from datetime import datetime

import threading


class FeatureStore:
    """
    Centralized storage for analytics features.

    Responsibilities:
    - Maintain per-symbol time series of features
    - Support incremental updates (append new row per symbol)
    - Provide historical queries for training, backtesting, analysis
    - incremental update(symbol, timestamp, features, pipeline_id)
    - latest(symbol), history(symbol, window/start/end)
    - materialize_for_training(symbol, end_ts, window, feature_cols, target_col)
     Thread-safe per-symbol operations (simple locks).
    """

    def __init__(self, base_path: Optional[str] = None):
        self._data: Dict[str, pd.DataFrame] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self.base_path = base_path

    def _lock(self, symbol: str) -> threading.Lock:
        if symbol not in self._locks:
            self._locks[symbol] = threading.Lock()
        return self._locks[symbol]

    def update(self, symbol: str, timestamp: datetime, features, pipeline_id: str = "v1"):
        """
        Incrementally update feature store with new computed features.

        Args:
            symbol (str): Asset symbol (e.g., "AAPL")
            timestamp (datetime): Market timestamp (e.g., bar close time)
            features (dict): Dict of feature_name -> value
        """

        ts = pd.to_datetime(timestamp)
        row = dict(features)
        row["_pipeline_id"] = pipeline_id
        row["_ts"] = ts

        df_row = pd.DataFrame([features], index=[pd.to_datetime(ts)])
        with self._lock(symbol):
            if symbol not in self._data or self._data[symbol].empty:
                self._data[symbol] = df_row
            else:
                df = pd.concat([self._data[symbol], df_row])
                df = df[~df.index.duplicated(keep="last")].sort_index()
                self._data[symbol] = df

    def latest(self, symbol: str, cols: Optional[List[str]] = None) -> Optional[pd.Series]:
        """
        Get the most recent feature row for a symbol.

        Args:
            symbol (str): Asset symbol

        Returns:
            pd.Series or None
        """
        df = self._data.get(symbol)
        if df is None or df.empty:
            return pd.Series(dtype=float)
        s = df.iloc[-1]
        return s[cols] if cols else s

    def history(
            self,
            symbol: str,
            window: Optional[int] = None,
            start: Optional[datetime] = None,
            end: Optional[datetime] = None,
            cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Query historical feature data.

        Args:
            symbol (str): Asset symbol
            window (int, optional): rolling window length (last N rows)
            start (datetime, optional): start time
            end (datetime, optional): end time

        Returns:
            pd.DataFrame: subset of features
        """
        df = self._data.get(symbol)
        if df is None or df.empty:
            return pd.DataFrame()

        # filter by time
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]

        # filter by window
        if window:
            df = df.tail(window)

        return df[cols].copy() if cols else df.copy()

    def available_symbols(self):
        """Return list of symbols currently stored."""
        return list(self._store.keys())

    def __repr__(self):
        return f"<FeatureStore symbols={len(self._store)}>"

    def materialize_for_training(self, symbol: str, end_ts, window: int,
                                 feature_cols: List[str], target_col: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Return X (features) and y (target) up to end_ts (inclusive), last `window` rows.
        Ensures point-in-time: no row > end_ts returned.
        """
        df = self.history(symbol, window=window+1, end=end_ts)  # +1 in case label uses next row
        if df.empty:
            return None, None
        if target_col not in df.columns:
            return None, None
        # For common next-bar target, we often use X = all rows except last, y = target[:-1]
        X = df[feature_cols].iloc[:-1]
        y = df[target_col].iloc[:-1].astype(int)
        if X.empty or y.empty:
            return None, None
        return X, y

    def persist_symbol(self, symbol: str, path: Optional[str] = None):
        path = path or os.path.join(self.base_path or ".", f"{symbol}.parquet")
        df = self._data.get(symbol)
        if df is None or df.empty:
            return
        df.reset_index().to_parquet(path, index=False)

    def load_symbol(self, symbol: str, path: str):
        df = pd.read_parquet(path)
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["_ts"])))
        with self._lock(symbol):
            self._data[symbol] = df