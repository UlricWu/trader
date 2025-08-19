# trader/visualization.py
# -*- coding: utf-8 -*-
# @File    : visualization.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/8/19

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trader.statistics as Stats


# ---------------------------
# Visualization Functions
# ---------------------------

def plot_daily_pnl(summary):
    """Daily and cumulative PnL per symbol."""
    fig = make_subplots(
        rows=len(summary) - 1, cols=2,
        subplot_titles=("Daily PnL", "Cumulative PnL")
    )

    row = 1
    for sym, e in summary.items():
        if sym == "account":
            continue
        daily_pnl = e["daily_pnl"]
        cum_pnl = daily_pnl.cumsum()

        fig.add_trace(
            go.Bar(x=daily_pnl.index, y=daily_pnl, name=f"{sym} Daily"),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=daily_pnl.index, y=cum_pnl, mode="lines", name=f"{sym} Cum"),
            row=row, col=2
        )
        row += 1

    fig.update_layout(height=500, width=1000, showlegend=False, template="plotly_white")
    return fig


def plot_drawdowns(summary):
    """Drawdown curves per symbol."""
    fig = go.Figure()
    for sym, e in summary.items():
        if sym == "account":
            continue
        equity = e["equity"]
        drawdown = equity / equity.cummax() - 1
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values * 100,
            mode="lines", name=sym, fill="tozeroy"
        ))

    fig.update_layout(
        title="Drawdowns (%)", yaxis_title="Drawdown (%)", xaxis_title="Date",
        hovermode="x unified", template="plotly_white"
    )
    return fig


def plot_heatmap(summary):
    """Monthly return heatmap per symbol."""
    symbols = [s for s in summary.keys() if s != "account"]
    fig = make_subplots(rows=1, cols=len(symbols), subplot_titles=symbols)

    for col, sym in enumerate(symbols, start=1):
        pivot_table = Stats._monthly_return_matrix(summary[sym]["returns"])
        heatmap = go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale="RdYlGn",
            zmin=-0.1, zmax=0.1,
            colorbar=dict(title="Return", x=1.0 + (0.05 * (col - 1)))
        )
        fig.add_trace(heatmap, row=1, col=col)

    fig.update_layout(
        height=400, width=300 * len(symbols),
        title="Monthly Return Heatmap", template="plotly_white"
    )
    return fig


def plot_price_trend_with_trades(df, bt, symbols):
    """Price lines with executed trades (markers)."""
    fig = go.Figure()
    for sym in symbols:
        price_data = df[df["ts_code"] == sym].set_index("trade_date").sort_index()
        fig.add_trace(go.Scatter(
            x=price_data.index, y=price_data["close"],
            mode="lines", name=sym
        ))

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
            x=df_trades.index, y=df_trades["price"], mode="markers",
            marker=dict(
                symbol=["triangle-up" if d == "BUY" else "triangle-down" for d in df_trades["direction"]],
                color=["red" if d == "BUY" else "green" for d in df_trades["direction"]],
                size=8
            ),
            customdata=df_trades[["quantity", "realized_pnl"]].values,
            hovertemplate="Price: %{y}<br>Qty: %{customdata[0]}<br>PnL: %{customdata[1]}<extra></extra>",
            name=f"{sym} Trades"
        )

    fig.update_layout(
        title="Price Trend with Executed Trades", xaxis_title="Date", yaxis_title="Price",
        legend=dict(orientation="h", y=1.05), hovermode="x unified", template="plotly_white"
    )
    return fig


def plot_pnl_distribution(summary):
    """Histogram of PnL across all symbols."""
    df = pd.DataFrame(dict(
        pnl=pd.concat([e["daily_pnl"] for sym, e in summary.items() if sym != "account"]),
        symbol=np.concatenate([[sym] * len(e["daily_pnl"]) for sym, e in summary.items() if sym != "account"])
    ))
    fig = px.histogram(df, x="pnl", color="symbol", nbins=50, marginal="box", title="PnL Distribution")
    fig.update_layout(template="plotly_white")
    return fig


def plot_win_loss(bt):
    """Pie chart win vs loss trades per symbol."""
    df_trades = pd.DataFrame([{
        "pnl": t.realized_pnl,
        "symbol": t.symbol
    } for t in bt.portfolio.transactions])
    df_trades["result"] = df_trades["pnl"].apply(lambda x: "Win" if x > 0 else "Loss")

    syms = df_trades["symbol"].unique().tolist()
    fig = make_subplots(rows=1, cols=len(syms), specs=[[{'type': 'domain'} for _ in syms]])

    for col, s in enumerate(syms):
        counts = df_trades[df_trades["symbol"] == s]["result"].value_counts()
        fig.add_trace(go.Pie(labels=counts.index, values=counts.values, title=f"{s} "), row=1, col=col + 1)

    fig.update_traces(textinfo="percent+label")
    fig.update_layout(template="plotly_white")
    return fig


def plot_rolling_sharpe(summary, window=30):
    """Rolling Sharpe ratio (first symbol)."""
    sym = [s for s in summary.keys() if s != "account"][0]
    rets = summary[sym]["returns"]
    roll_sharpe = rets.rolling(window).mean() / rets.rolling(window).std()
    fig = px.line(roll_sharpe, title=f"Rolling {window}-day Sharpe Ratio")
    fig.update_layout(template="plotly_white")
    return fig


def plot_symbol_contribution(bt):
    """PnL contribution by symbol."""
    df_trades = pd.DataFrame([{
        "pnl": t.realized_pnl,
        "symbol": t.symbol
    } for t in bt.portfolio.transactions])
    contrib = df_trades.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
    fig = px.bar(contrib, title="PnL Contribution by Symbol")
    fig.update_layout(template="plotly_white")
    return fig
