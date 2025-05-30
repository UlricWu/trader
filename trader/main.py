#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : main.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/20 22:39

# main.py
import pandas as pd
from backtest_engine import Backtest

if __name__ == "__main__":
    # df = pd.read_csv("your_data.csv")  # Must contain date + close columns
    # df["date"] = pd.to_datetime(df["date"])

    # 1. Load data from SQLite
    from data import db

    code = "000001.SZ"

    df = db.extract_table(day="20250205", start_day='20240601', ts_code=[code])
    df = db.load_and_normalize_data(df)

    engine = Backtest(data=df)
    engine.run()
