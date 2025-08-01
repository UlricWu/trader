#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : visualization.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/1 11:13
# dashboard.py
# from typing import Dict
## trader/visualization.py
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_and_drawdown(equity_df: pd.DataFrame, title: str = "Equity Curve"):
    """
    Plots equity curve and drawdown.
    equity_df must contain 'equity' column and datetime index.
    """
    if equity_df.empty or "equity" not in equity_df.columns:
        print("No equity data available.")
        return

    # Compute drawdown
    cummax = equity_df["equity"].cummax()
    drawdown = equity_df["equity"] / cummax - 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    # Plot equity
    ax1.plot(equity_df.index, equity_df["equity"], label="Equity", color="blue")
    ax1.set_title(title)
    ax1.set_ylabel("Equity Value")
    ax1.legend()
    ax1.grid(True)

    # Plot drawdown
    ax2.fill_between(equity_df.index, drawdown, color="red", alpha=0.3)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
# def plot_per_symbol_equity(symbol_equity_dfs: Dict[str, pd.DataFrame]):
#     """
#     Plots equity curve for each symbol.
#     """
#     num_symbols = len(symbol_equity_dfs)
#     fig, ax = plt.subplots(figsize=(12, 6))
#
#     for symbol, df in symbol_equity_dfs.items():
#         if df.empty:
#             continue
#         ax.plot(df.index, df["equity"], label=symbol)
#
#     ax.set_title("Per-Symbol Equity Curves")
#     ax.set_ylabel("Equity Value")
#     ax.set_xlabel("Date")
#     ax.legend()
#     ax.grid(True)
#     plt.tight_layout()
#     # plt.show()
#     return fig

# import streamlit as st
# import matplotlib.pyplot as plt
# import pandas as pd
# from trader.config import load_settings
#
# from performance import PerformanceAnalyzer
# from backtest_engine import Backtest  # Assuming you have this
# # from visualization import plot_equity_and_drawdown, plot_per_symbol_equity
#
# # trader/visualization.py
# import matplotlib.pyplot as plt
# import pandas as pd
#
# def plot_per_symbol_equity(symbol_equity_dfs: Dict[str, pd.DataFrame]):
#     """
#     Plots equity curve for each symbol.
#     """
#     num_symbols = len(symbol_equity_dfs)
#     fig, ax = plt.subplots(figsize=(12, 6))
#
#     for symbol, df in symbol_equity_dfs.items():
#         if df.empty:
#             continue
#         ax.plot(df.index, df["equity"], label=symbol)
#
#     ax.set_title("Per-Symbol Equity Curves")
#     ax.set_ylabel("Equity Value")
#     ax.set_xlabel("Date")
#     ax.legend()
#     ax.grid(True)
#     plt.tight_layout()
#     plt.show()
#
# def plot_equity_and_drawdown(equity_df: pd.DataFrame, title: str = "Equity Curve"):
#     """
#     Plots equity curve and drawdown.
#     equity_df must contain 'equity' column and datetime index.
#     """
#     if equity_df.empty or "equity" not in equity_df.columns:
#         print("No equity data available.")
#         return
#
#     # Compute drawdown
#     cummax = equity_df["equity"].cummax()
#     drawdown = equity_df["equity"] / cummax - 1.0
#
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
#                                    gridspec_kw={'height_ratios': [2, 1]})
#
#     # Plot equity
#     ax1.plot(equity_df.index, equity_df["equity"], label="Equity", color="blue")
#     ax1.set_title(title)
#     ax1.set_ylabel("Equity Value")
#     ax1.legend()
#     ax1.grid(True)
#
#     # Plot drawdown
#     ax2.fill_between(equity_df.index, drawdown, color="red", alpha=0.3)
#     ax2.set_ylabel("Drawdown")
#     ax2.set_xlabel("Date")
#     ax2.grid(True)
#
#     plt.tight_layout()
#     plt.show()
#
#
# def show_equity_curve(performance: PerformanceAnalyzer):
#     st.subheader("Account Equity Curve")
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
#                                    gridspec_kw={'height_ratios': [2, 1]})
#
#     equity_df = performance.equity_df
#     cummax = equity_df["equity"].cummax()
#     drawdown = equity_df["equity"] / cummax - 1.0
#
#     ax1.plot(equity_df.index, equity_df["equity"], label="Equity", color="blue")
#     ax1.set_ylabel("Equity Value")
#     ax1.legend()
#     ax1.grid(True)
#
#     ax2.fill_between(equity_df.index, drawdown, color="red", alpha=0.3)
#     ax2.set_ylabel("Drawdown")
#     ax2.set_xlabel("Date")
#     ax2.grid(True)
#
#     plt.tight_layout()
#     st.pyplot(fig)
#
#
# def show_per_symbol_equity(performance: PerformanceAnalyzer):
#     st.subheader("Per-Symbol Equity Curves")
#     symbol_dfs = performance.portfolio.symbol_equity_dfs
#     fig, ax = plt.subplots(figsize=(12, 6))
#
#     for symbol, df in symbol_dfs.items():
#         if df.empty:
#             continue
#         ax.plot(df.index, df["equity"], label=symbol)
#
#     ax.set_ylabel("Equity Value")
#     ax.set_xlabel("Date")
#     ax.legend()
#     ax.grid(True)
#     plt.tight_layout()
#     st.pyplot(fig)
#
#
# def show_metrics(performance: PerformanceAnalyzer):
#     st.subheader("Performance Summary")
#
#     account_metrics = performance.account_summary()
#     st.write("### Account-Level Metrics")
#     st.dataframe(pd.DataFrame(account_metrics.items(), columns=["Metric", "Value"]))
#

#
#
# def main():
#     st.title("Trading Strategy Performance Dashboard")
#
#     # Run or Load Backtest
#     # if st.button("Run Backtest"):
#     #     bt = Backtest()
#     #     bt.run()
#     #     performance = bt.performance
#     # else:
#     #     st.info("Click 'Run Backtest' to load data.")
#     #     st.stop()
#     # 1. Load data from SQLite
#     from data import db
#
#     code = "000001.SZ"
#
#     df = db.extract_table(day="20250205", start_day='20240601', ts_code=[code])
#     data = db.load_and_normalize_data(df)
#     # data = load_data(
#     #     adjustment=settings.data.price_adjustment,
#     #     symbols=settings.trading.symbol_list
#     # )
#     settings = load_settings()
#     bt = Backtest(data, settings=settings)
#     bt.run()
#     performance = PerformanceAnalyzer(portfolio=bt.portfolio)
#
#     # Show Visuals
#     show_equity_curve(performance)
#     show_per_symbol_equity(performance)
#     show_metrics(performance)
#
#
# if __name__ == "__main__":
#     main()
