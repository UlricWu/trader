#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : dashboard.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/15
import time
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd
import plotly.express as px
import streamlit as st

from trader.config import load_settings
from trader.backtest_engine import Backtest
from trader.performance import PerformanceAnalyzer
from data import db


def main():
    st.set_page_config(layout="wide", page_title="Trading Performance Dashboard")
    st.title("📈 Trading Performance Dashboard")

    # Sidebar settings
    st.sidebar.header("⚙️ Dashboard Settings")
    live_mode = st.sidebar.checkbox("Enable Live Mode", value=False)
    refresh_sec = st.sidebar.slider("Refresh Interval (sec)", 5, 60, 10)

    settings = load_settings()

    # Symbol selection
    all_codes = db.extract_codes(database=settings.data.database)
    show_symbols = st.sidebar.multiselect(
        "Select symbols to display", options=list(all_codes),
        default=list(all_codes[:3])
    )
    symbols = [s[0] for s in show_symbols]

    # Date range selection
    st.sidebar.subheader("Date Range")
    start_date = st.sidebar.date_input("Start Date", value=settings.data.start_day)
    end_date = st.sidebar.date_input("End Date", value=settings.data.end_day)

    if st.button("Run Backtest"):
        df = db.extract_table(
            database=settings.data.database,
            start_day=start_date.strftime("%Y%m%d"),
            end_day=end_date.strftime("%Y%m%d"),
            ts_code=symbols
        )
        data = db.load_and_normalize_data(df)

        bt = Backtest(data, settings=settings)
        bt.run()
        performance = PerformanceAnalyzer(portfolio=bt.portfolio)

        # Portfolio Summary
        st.header("Portfolio Summary Metrics")
        summary = performance.summary()

        st.subheader("Performance Table")
        df_table = performance._stats_table(summary)
        st.table(df_table)

        # Equity Curves
        st.subheader("Equity Curves")
        eq_df = performance._equity_df(summary)
        fig = px.line(eq_df, title='Equity Curves')
        st.plotly_chart(fig, use_container_width=True)

        # Drawdowns
        st.subheader("Drawdowns")
        fig, ax = plt.subplots(figsize=(12, 3))
        performance._plot_drawdown(ax, summary)
        st.pyplot(fig)

        # Monthly Returns
        st.subheader("Monthly Returns Heatmaps")
        for sym in symbols:
            s = summary[sym]
            fig, ax = plt.subplots(figsize=(12, 2))
            performance._plot_monthly_returns(ax, s)
            st.pyplot(fig)

    # Trend Plots with Buy/Sell markers
        st.subheader("Price Trend with Executed Trades")
        for sym in symbols:
            df_symbol = df[df["ts_code"] == sym].copy()
            df_symbol["trade_date"] = pd.to_datetime(df_symbol["trade_date"])
            df_symbol.set_index("trade_date", inplace=True)
            df_symbol.sort_index(inplace=True)

            trades = [t for t in bt.portfolio.transactions if t.symbol == sym]
            if trades:
                df_trades = pd.DataFrame([{
                    "direction": t.direction,
                    "price": t.price,
                    "quantity": t.quantity,
                    "realized_pnl": t.realized_pnl,
                    "date": t.date
                } for t in trades])
                df_trades["date"] = pd.to_datetime(df_trades["date"])
                df_trades.set_index("date", inplace=True)
            else:
                df_trades = pd.DataFrame(columns=["direction", "price", "quantity", "realized_pnl"])

            fig_trend = plot_price_with_trades_plotly(df_symbol, df_trades, sym)
            st.plotly_chart(fig_trend, use_container_width=True)


def plot_price_with_trades_plotly(df_price: pd.DataFrame, trades: pd.DataFrame, symbol: str):
    """
    Plot price trend with executed trades using Plotly with hover tooltips.
    """
    fig = px.line(df_price, y="close", x=df_price.index, title=f"{symbol} Price Trend with Trades",
                  labels={"x": "Date", "close": "Price"})

    # Add Buy trades
    buy_trades = trades[trades["direction"] == "BUY"]
    if not buy_trades.empty:
        fig.add_scatter(
            x=buy_trades.index,
            y=buy_trades["price"],
            mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=12),
            name="BUY",
            hovertemplate=
            "Price: %{y}<br>Qty: %{customdata[0]}<br>Realized PnL: %{customdata[1]}<extra></extra>",
            customdata=buy_trades[["quantity", "realized_pnl"]].values
        )

    # Add Sell trades
    sell_trades = trades[trades["direction"] == "SELL"]
    if not sell_trades.empty:
        fig.add_scatter(
            x=sell_trades.index,
            y=sell_trades["price"],
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=12),
            name="SELL",
            hovertemplate=
            "Price: %{y}<br>Qty: %{customdata[0]}<br>Realized PnL: %{customdata[1]}<extra></extra>",
            customdata=sell_trades[["quantity", "realized_pnl"]].values
        )

    fig.update_layout(xaxis_title="Date", yaxis_title="Price", legend=dict(orientation="h", y=1.05))
    return fig


if __name__ == "__main__":
    main()
