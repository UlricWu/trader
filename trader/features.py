#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : features.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/10 11:20
#!features.py

import pandas as pd

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_1d"] = df["close"].pct_change()
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_ratio"] = df["sma_5"] / df["sma_20"]
    df = df.dropna()
    return df
