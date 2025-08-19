#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : dashboard.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/15

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trader.config import load_settings
from trader.backtest_engine import Backtest
from trader.performance import PerformanceAnalyzer
from data import db
from plotly.subplots import make_subplots
import trader.statistics as Stats
from trader.rulestrategy import MLStrategy
# ---------------------------
# Helper Functions
# ---------------------------

from utilts.logs import logs


def plot_drawdowns(summary):
    fig = go.Figure()
    for sym in summary.keys():
        if sym == 'account':
            continue
        equity = summary[sym]["equity"]
        drawdown = equity / equity.cummax() - 1
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values * 100,
            mode='lines', name=sym, fill='tozeroy'
        ))
    fig.update_layout(
        title="Drawdowns (%)",
        yaxis_title="Drawdown (%)",
        xaxis_title="Date",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig


def plot_heatmap(summary):
    symbols = [s for s in summary.keys() if s != "account"]

    fig = make_subplots(
        rows=1,
        cols=len(symbols),
        subplot_titles=symbols
    )

    for col, sym in enumerate(symbols, start=1):
        pivot_table = Stats._monthly_return_matrix(summary[sym]["returns"])
        heatmap = go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale="RdYlGn",
            zmin=-0.1,  # adjust scale for returns
            zmax=0.1,
            colorbar=dict(title="Return", x=1.0 + (0.05 * (col - 1)))
        )
        fig.add_trace(heatmap, row=1, col=col)

    fig.update_layout(
        height=400,
        width=300 * len(symbols),
        title="Monthly Return Heatmap per Symbol",
        showlegend=False,
        template="plotly_white"
    )

    return fig


def plot_price_trend_with_trades(df, bt, symbols):
    fig = go.Figure()
    for sym in symbols:
        # Price line
        price_data = df[df["ts_code"] == sym].set_index("trade_date").sort_index()
        fig.add_trace(go.Scatter(
            x=price_data.index, y=price_data["close"],
            mode='lines', name=sym
        ))

        # Trade markers
        trades = [t for t in bt.portfolio.transactions if t.symbol == sym]
        if not trades:
            continue

        df_trades = pd.DataFrame([{
            "direction": t.direction,
            "price": round(t.price, 2),
            "quantity": t.quantity,
            "realized_pnl": round(t.realized_pnl, 2),
            "date": pd.to_datetime(t.date)
        } for t in trades]).set_index("date")

        fig.add_scatter(
            x=df_trades.index,
            y=df_trades["price"],
            mode="markers",
            marker=dict(
                symbol=["triangle-up" if d == "BUY" else "triangle-down" for d in df_trades["direction"]],
                color=["red" if d == "BUY" else "green" for d in df_trades["direction"]],
                size=8
            ),
            name='',
            hovertemplate=(
                "Price: %{y}<br>"
                "Qty: %{customdata[0]}<br>"
                "Realized PnL: %{customdata[1]}<extra></extra>"
            ),
            customdata=df_trades[["quantity", "realized_pnl"]].values
        )

    fig.update_layout(
        title="Price Trend with Buy/Sell Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(orientation="h", y=1.05),
        hovermode="x unified",
        template="plotly_white"
    )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='LightGray')
    return fig


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

        st.subheader("Performance Table")
        st.table(performance._stats_table(summary))

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


if __name__ == "__main__":
    main()
