#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : dashboard.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/15
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from matplotlib import pyplot as plt

from trader.config import load_settings
from trader.backtest_engine import Backtest
from trader.performance import PerformanceAnalyzer
from data import db
from plotly.subplots import make_subplots
import trader.statistics as Stats
# dashboard.py
from trader.visualization import (
    plot_daily_pnl, plot_drawdowns, plot_heatmap,
    plot_price_trend_with_trades, plot_pnl_distribution,
    plot_win_loss, plot_rolling_sharpe, plot_symbol_contribution
)
from trader.strategy import MLStrategy
# ---------------------------
# Helper Functions
# ---------------------------

from utilts.logs import logs


# ---------------------------
# Main Dashboard
# ---------------------------

def main():
    st.set_page_config(layout="wide", page_title="Trading Performance Dashboard")
    st.title("📈 Trading Performance Dashboard")

    # Load settings
    settings = load_settings()

    # Sidebar controls
    st.sidebar.header("⚙️ Dashboard Settings")
    live_mode = st.sidebar.checkbox("Enable Live Mode", value=False)
    refresh_sec = st.sidebar.slider("Refresh Interval (sec)", 5, 60, 10)

    # Symbol selection
    all_codes = db.extract_codes(database=settings.data.database)
    symbols = st.sidebar.multiselect(
        "Select symbols to display", options=all_codes,
        default=all_codes[:2]
    )

    # Date range
    st.sidebar.subheader("Date Range")
    start_date = st.sidebar.date_input("Start Date", value=settings.data.start_day)
    end_date = st.sidebar.date_input("End Date", value=settings.data.end_day)

    # Load data
    df = db.extract_table(
        database=settings.data.database,
        start_day=start_date.strftime("%Y%m%d"),
        end_day=end_date.strftime("%Y%m%d"),
        ts_code=symbols
    )
    data = db.load_and_normalize_data(df)

    # Price overview chart
    price_pivot = data[['symbol', 'date', 'close']].pivot_table(index='date', columns='symbol', values='close')
    fig = px.line(price_pivot, title="Price Overview")
    fig.update_layout(hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.info("Dashboard updates automatically when you change symbols or date range.")

    # Backtest and performance
    if st.button("Run Backtest"):
        bt = Backtest(data=data, settings=settings)
        # bt = Backtest(data=data, settings=settings, strategy_class=MLStrategy)

        bt.run()
        performance = PerformanceAnalyzer(portfolio=bt.portfolio)
        summary = performance.summary()

        # Portfolio Summary Table
        st.header("Portfolio Summary Metrics")

        # Daily PnL TradesViz-style
        st.subheader("Daily PnL")

        daily_pnl_fig = plot_daily_pnl(summary)
        st.plotly_chart(daily_pnl_fig, use_container_width=True)

        st.subheader("Performance Table")
        st.table(performance._stats_table(summary))

        with pd.option_context('display.max_rows', None, 'display.max_columns',
                               None):  # more options can be specified also
            print(performance._stats_table(summary))

        # Equity Curves
        st.subheader("Equity Curves")
        df_equity = performance._equity_curve_df(summary)
        fig = px.line(df_equity, title="Equity Curves")
        fig.update_layout(hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        # Drawdowns
        st.subheader("Drawdowns")
        st.plotly_chart(plot_drawdowns(summary), use_container_width=True)

        # Monthly Returns
        st.subheader("Monthly Returns Heatmaps")
        st.plotly_chart(plot_heatmap(summary), use_container_width=True)

        # Price Trend with Trades
        st.subheader("Price Trend with Executed Trades")
        st.plotly_chart(plot_price_trend_with_trades(df, bt, symbols), use_container_width=True)

        # Extra Analytics
        st.subheader("PnL Distribution")
        st.plotly_chart(plot_pnl_distribution(summary), use_container_width=True)

        st.subheader("Win/Loss Breakdown")
        st.plotly_chart(plot_win_loss(bt), use_container_width=True)

        st.subheader("Rolling Sharpe Ratio")
        st.plotly_chart(plot_rolling_sharpe(summary), use_container_width=True)

        st.subheader("Symbol Contribution")
        st.plotly_chart(plot_symbol_contribution(bt), use_container_width=True)


if __name__ == "__main__":
    main()
