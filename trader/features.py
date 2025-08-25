#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : features.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/22 18:13

import pandas as pd
from trader.events import FeatureEvent


class FeatureEngineer:
    """
    Computes rolling technical indicators (simple version).
    """

    def __init__(self, lookbacks=None):
        self.lookbacks = lookbacks or {"MA5": 5, "MA10": 10, "Volatility20": 20}
        self.data = {}

    def on_market(self, event):

        symbol = event.symbol
        # print(event)
        price = event.close
        ts = event.datetime

        if symbol not in self.data:
            self.data[symbol] = pd.DataFrame(columns=["price"])
        self.data[symbol].loc[ts] = price

        df = self.data[symbol]
        features = {"close": price}

        for name, window in self.lookbacks.items():
            if "MA" in name:
                features[name] = df["price"].rolling(window).mean().iloc[-1]
            elif "Volatility" in name:
                features[name] = df["price"].pct_change().rolling(window).std().iloc[-1]

        features["Return_1d"] = df["price"].pct_change().iloc[-1]
        return FeatureEvent(symbol=symbol, datetime=ts, features=features)
