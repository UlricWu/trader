#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : backtest_pipeline.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/26 20:54

# backtest_pipeline.py
# backtest_pipeline.py
from trader.config import load_settings
from trader.backtest_engine import Backtest
from trader.performance import PerformanceAnalyzer
# from trader.visualization import plot_equity_and_drawdown, plot_per_symbol_equity

def send_slack_notification(summary: dict):
    msg = f"Backtest Completed. Return: {summary['total_return'] * 100:.2f}%"


def backtest(settings):
    # 1. Load data from SQLite
    from data import db

    code = "000001.SZ"

    df = db.extract_table(end_day="20250205", start_day='20240601', ts_code=[code])
    data = db.load_and_normalize_data(df)
    # data = load_data(
    #     adjustment=settings.data.price_adjustment,
    #     symbols=settings.trading.symbol_list
    # )

    bt = Backtest(data, settings=settings)
    bt.run()
    perf = PerformanceAnalyzer(portfolio=bt.portfolio)
    print(perf.summary())
    # plot_equity_and_drawdown(bt.portfolio.equity_df)
    # plot_per_symbol_equity(bt.portfolio.symbol_equity_dfs)
    # print(bt.portfolio.equity_df)
    # bt.plot_equity_curve()


if __name__ == "__main__":
    settings = load_settings()
    backtest(settings)
