#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : dashboard.py.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/1 17:12
# dashboard.py
import time
from typing import Dict, List

import matplotlib.pyplot as plt

from trader.config import load_settings
from trader.backtest_engine import Backtest
from trader.performance import PerformanceAnalyzer

import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
import pandas as pd


def run_and_display(settings):
    pass


def main():
    st.set_page_config(layout="wide", page_title="Trading Performance Dashboard")
    st.title("📈 Trading Performance Dashboard")

    st.sidebar.header("⚙️ Dashboard Settings")
    live_mode = st.sidebar.checkbox("Enable Live Mode", value=False)
    refresh_sec = st.sidebar.slider("Refresh Interval (sec)", 5, 60, 10)
    # if live_mode:
    #     st.success(f"Live Mode Enabled — refreshing every {refresh_sec} sec.")
    #     time.sleep(refresh_sec)

    settings = load_settings()
    st.info("Click 'Run Backtest' to start.")

    from data import db

    all_codes = db.extract_codes(database=settings.data.database)
    # --- Sidebar controls ---
    st.sidebar.header("Controls")
    show_symbols = st.sidebar.multiselect(
        "Select symbols to display", options=list(all_codes),
        default=list(all_codes[:3])
    )

    # print(list(show_symbols))
    symbols = [s[0] for s in show_symbols]
    print(symbols)

    if st.button("Run Backtest"):

        #
        #
        df = db.extract_table(database=settings.data.database, end_day=settings.data.end_day,
                              start_day=settings.data.start_day, ts_code=symbols)
        data = db.load_and_normalize_data(df)

        print(data.head())
        #
        bt = Backtest(data, settings=settings)
        bt.run()
        performance = PerformanceAnalyzer(portfolio=bt.portfolio)
        #
        #
        #
        # # --- Summary metrics ---
        st.header("Portfolio Summary Metrics")
        summary = performance.summary()

        # Filter symbols
        symbols_to_plot = show_symbols.copy()
        # if show_account:
        #     symbols_to_plot.insert(0, "account")

        # --- Stats table ---
        # if show_stats_table:
        st.subheader("Performance Table")
        df_table = performance._stats_table(summary)
        st.table(df_table)

        # --- Equity curve ---
        st.subheader("Equity Curves")

        df = performance._equity_df(summary)

        # Create a Plotly line chart with multiple lines
        fig = px.line(df, title='Equity Curves')

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # --- Drawdowns ---
        st.subheader("Drawdowns")
        fig, ax = plt.subplots(figsize=(12, 3))
        performance._plot_drawdown(ax, summary)
        st.pyplot(fig)

        # --- Monthly returns heatmaps ---
        st.subheader("Monthly Returns Heatmaps")
        for sym in symbols:
            s = summary[sym]
            fig, ax = plt.subplots(figsize=(12, 2))
            performance._plot_monthly_returns(ax, s)
            st.pyplot(fig)


if __name__ == "__main__":
    main()
    # python -m streamlit run dashboard.py
