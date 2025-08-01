#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : dashboard.py.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/1 17:12
# dashboard.py
from typing import Dict, List

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from trader.config import load_settings
from trader.backtest_engine import Backtest
from trader.performance import PerformanceAnalyzer


def plot_equity_drawdown(equity_df: pd.DataFrame, title: str = "Equity Curve") -> plt.Figure:
    cummax = equity_df["equity"].cummax()
    drawdown = equity_df["equity"] / cummax - 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(equity_df.index, equity_df["equity"], label="Equity", color="blue")
    ax1.set_title(title)
    ax1.set_ylabel("Equity")
    ax1.grid(True)
    ax1.legend()

    ax2.fill_between(equity_df.index, drawdown, color="red", alpha=0.3)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True)

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(layout="wide", page_title="Trading Performance Dashboard")
    st.title("📈 Trading Performance Dashboard")

    settings = load_settings()
    st.info("Click 'Run Backtest' to start.")

    if st.button("Run Backtest"):
        # 1. Load data from SQLite
        from data import db

        df = db.extract_table(name=settings.data.name, end_day=settings.data.end_day, start_day=settings.data.start_day,
                              ts_code=settings.data.ts_code)
        data = db.load_and_normalize_data(df)

        bt = Backtest(data, settings=settings)
        bt.run()
        performance = PerformanceAnalyzer(portfolio=bt.portfolio)

        # Create Tabs for layout clarity
        tab1, tab2 = st.tabs(["📊 Account Overview", "📈 Per-Symbol Performance"])

        # Tab 1: Account-Level Overview
        with tab1:
            st.subheader("📊 Account-Level Metrics")
            metrics = performance.account_summary()

            # Layout: Left stats, right plot
            left_col, right_col = st.columns([1, 2])  # Ratio 1:2

            with left_col:
                st.subheader("📊  Stat")
                for m, value in metrics.items():
                    if abs(value) > 1:
                        st.metric(m, round(value, 2))
                    else:
                        st.metric(m, f"{value:.2%}")

            with right_col:
                st.subheader("📉 Equity Curve + Drawdown")
                fig = plot_equity_drawdown(performance.equity_df, title="Equity & Drawdown")
                st.pyplot(fig)

            st.divider()
            st.markdown("---")
        # Tab 2: Per-Symbol Metrics + Trade Plots
        with tab2:
            st.subheader("📈 Per-Symbol Performance")

            # Per-Symbol Section Below
            left_col, right_col = st.columns([1, 2])  # Ratio 1:2
            with left_col:
                st.subheader("📊  Stat")
                symbol_metrics = performance.symbol_summary()
                st.table(pd.DataFrame(symbol_metrics))

            with right_col:
                for symbol in symbol_metrics.keys():

                    price_df = bt.data_handler.get_symbol_bars(symbol)
                    trades = [t for t in performance.portfolio.transactions if t.symbol == symbol]

                    if price_df is not None and not price_df.empty:
                        fig = plot_candlestick_with_trades(symbol, price_df, trades)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No price data for {symbol}")


import plotly.graph_objects as go
import pandas as pd


def plot_candlestick_with_trades(
        symbol: str,
        price_data: pd.DataFrame,
        trades: list
) -> go.Figure:
    """
    Create candlestick chart with Buy/Sell markers using Plotly.

    Args:
        symbol: Symbol name.
        price_data: DataFrame with ['open', 'high', 'low', 'close'], indexed by date.
        trades: List of trade objects with symbol, date, price, quantity.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=price_data['date'],
        open=price_data['open'],
        high=price_data['high'],
        low=price_data['low'],
        close=price_data['close'],
        name='Candlestick'
    ))

    # Trades
    buy_trades = [t for t in trades if t.symbol == symbol and t.quantity > 0]
    sell_trades = [t for t in trades if t.symbol == symbol and t.quantity < 0]

    if buy_trades:
        fig.add_trace(go.Scatter(
            x=[t.date for t in buy_trades],
            y=[t.price for t in buy_trades],
            mode='markers',
            marker=dict(symbol='triangle-up', color='green', size=10),
            name='Buy'
        ))

    if sell_trades:
        fig.add_trace(go.Scatter(
            x=[t.date for t in sell_trades],
            y=[t.price for t in sell_trades],
            mode='markers',
            marker=dict(symbol='triangle-down', color='red', size=10),
            name='Sell'
        ))

    fig.update_layout(
        title=f"{symbol} Candlestick Chart with Trades",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600
    )

    return fig


def method_name(data_handler, performance, symbol):
    df = data_handler.get_symbol_bars(symbol)
    trades = [t for t in performance.portfolio.transactions if t.symbol == symbol]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df['date'], df["close"], label="Close Price", color="black", linewidth=1.5)
    # Optional: show open price line for reference
    # ax.plot(df['date'], df["open"], label="Open Price", color="blue", linestyle="--", alpha=0.6)
    # Trade markers
    buy_dates = [t.date for t in trades if t.quantity > 0]
    buy_prices = [t.price for t in trades if t.quantity > 0]
    sell_dates = [t.date for t in trades if t.quantity < 0]
    sell_prices = [t.price for t in trades if t.quantity < 0]
    ax.scatter(buy_dates, buy_prices, color="green", marker="^", label="Buy", s=10)
    ax.scatter(sell_dates, sell_prices, color="red", marker="v", label="Sell", s=10)
    ax.set_title(f"Live Trading Plot: {symbol}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend(loc="best")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    main()
