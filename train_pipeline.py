# train_pipeline.py
import os

import numpy as np
import pandas as pd

from trader.backtest_engine import Backtest

from trader.config import load_settings

from trader.performance import PerformanceAnalyzer
from trader.strategy import MLStrategy
# from trader.strategy import MLStrategy


def train():
    settings = load_settings()

    """Train pipeline in batch mode."""
    print("[Train Pipeline Started]")

    from data import db

    codes = ["000001.SZ", "000002.SZ"]

    df = db.extract_table(database='db/tutorial.db', end_day="20250205", start_day='20241001', ts_code=codes)
    data = db.load_and_normalize_data(df)

    bt = Backtest(data=data, settings=settings)
    # bt = Backtest(data=data, settings=settings, strategy_class=MLStrategy)

    bt.run()
    # print(bt.summary())
    perf = PerformanceAnalyzer(portfolio=bt.portfolio)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df)

    print(perf.summary())
    perf.plot()

    print("🏁 All symbol models trained and saved.")


if __name__ == "__main__":
    # your injected settings

    train()
