from typing import Tuple, Any

import pandas as pd
import numpy as np

DECIMALS = 4
YEAR = 252
MONTHLY = 12
WEEKLY = 52


def round_float(x: float) -> float:
    return round(x, DECIMALS)


def clean_zero_equity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace zeros with NaN to avoid divide-by-zero in returns.
    """
    return df.replace(0, pd.NA)


def aggregate_returns(returns, convert_to):
    """
    Aggregates returns by day, week, month, or year.
    """

    def cumulate_returns(x):
        return np.exp(np.log(1 + x).cumsum()).iloc[-1] - 1

    if convert_to == 'weekly':
        return returns.groupby(
            [lambda x: x.year,
             lambda x: x.month,
             lambda x: x.isocalendar()[1]]).apply(cumulate_returns)
    elif convert_to == 'monthly':
        return returns.groupby(
            [lambda x: x.year, lambda x: x.month]).apply(cumulate_returns)
    elif convert_to == 'yearly':
        return returns.groupby(
            [lambda x: x.year]).apply(cumulate_returns)
    else:
        ValueError('convert_to must be weekly, monthly or yearly')


def create_cagr(equity, periods=YEAR):
    """
    Calculates the Compound Annual Growth Rate (CAGR)
    for the portfolio, by determining the number of years
    and then creating a compound annualised rate based
    on the total return.

    Parameters:
    equity - A pandas Series representing the equity curve.
    periods - Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    """
    years = len(equity) / float(periods)
    cagr = (equity.iloc[-1] ** (1.0 / years)) - 1.0

    return round_float(cagr)


def create_sortino_ratio(returns, periods=YEAR):
    """
    Create the Sortino ratio for the strategy, based on a
    benchmark of zero (i.e. no risk-free rate information).

    Parameters:
    returns - A pandas Series representing period percentage returns.
    periods - Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    """
    return np.sqrt(periods) * (np.mean(returns)) / np.std(returns[returns < 0])


def _ensure_series(data) -> pd.DataFrame:
    """Ensure input is a DataFrame; convert Series to DataFrame."""
    if isinstance(data, pd.Series):
        return data.dropna()
    elif isinstance(data, pd.DataFrame):
        return data.squeeze().dropna()
    else:
        raise TypeError(f"Input must be a pandas DataFrame or Series, got {type(data)}")


def daily_returns(equity) -> pd.DataFrame:
    """Compute daily returns from equity curve."""
    equity = _ensure_series(equity)
    return round_float(equity.dropna().pct_change().fillna(0))


def aggregated_daily_pnl(equity) -> pd.Series:
    """
    Return aggregated daily PnL series.
    """
    equity = _ensure_series(equity)

    pnl = equity.diff().dropna()
    daily_pnl = pnl.groupby(pnl.index.date).sum()
    daily_pnl.index = pd.to_datetime(daily_pnl.index)
    return daily_pnl


def _monthly_return_matrix(returns: pd.DataFrame):
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_returns_df = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1).round(2).to_frame(
        "Return")
    monthly_returns_df["Year"] = monthly_returns_df.index.year
    monthly_returns_df["Month"] = monthly_returns_df.index.strftime("%b")
    pivot_table = monthly_returns_df.pivot(index="Year", columns="Month", values="Return")
    pivot_table = pivot_table.reindex(columns=month_order)
    return pivot_table


def annual_returns(equity) -> float:
    daily = daily_returns(equity)
    # Calculate the compounded return over the entire period
    compounded_return = (1 + daily).prod() - 1

    # Determine the total number of periods in your data
    total_periods = len(daily)

    # Calculate the annualized return
    annualized_return = (1 + compounded_return) ** (YEAR / total_periods) - 1

    return round_float(annualized_return)


# def cagr(equity: pd.Series) -> float:
#     years = (equity.index[-1] - equity.index[0]).days / 365.25
#     return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

def volatility(equity) -> float:
    daily = daily_returns(equity)
    volatility = daily.std() * (YEAR ** 0.5),
    return round_float(volatility[0])


def sharpe_ratio(equity, risk_free_rate=0.0) -> float:
    daily = daily_returns(equity)
    excess = daily - risk_free_rate / YEAR
    sharpe = round(excess.mean() / excess.std() * (YEAR ** 0.5), 4)
    return sharpe  # pd.series->float


def cumulative_returns(equity) -> float:
    """Compute cumulative returns from equity curve."""
    equity = _ensure_series(equity)
    daily_ret = daily_returns(equity)
    cum = (1 + daily_ret).cumprod() - 1
    return round_float(cum)


# from __future__ import annotations
from typing import Tuple, Union
import pandas as pd
import numpy as np


def max_drawdown_and_duration(e):
    """
    Compute maximum drawdown (negative float) and its duration in days.

    If `e` is a Series -> returns (mdd_float, duration_days_int).
    If `e` is a DataFrame -> returns (mdd_series, duration_series) aligned to columns.

    Assumptions:
      - Index represents time; function will try to convert to DatetimeIndex and sort.
      - Values are equity/wealth levels (not returns).
      - NaNs are ignored when possible.

    Raises:
      TypeError: if input is not a Series/DataFrame or index cannot be converted to datetime.
    """

    s = _ensure_series(e)
    cummax = s.cummax()
    dd = s.div(cummax) - 1.0

    mdd = float(dd.min())  # negative number (e.g., -0.23 for -23%)
    trough_ts = dd.idxmin()

    # restrict to data up to trough to find peak before trough
    pre = s.loc[:trough_ts].dropna()
    if pre.empty:
        return mdd, 0
    peak_ts = pre.idxmax()

    duration_days = int((trough_ts - peak_ts).days)
    return mdd, max(duration_days, 0)

# Example usage
# if __name__ == "__main__":
#     dates = pd.date_range("2024-01-01", periods=10, freq="D")
#     eq_series = pd.Series([100, 102, 101, 99, 98, 97, 99, 100, 101, 103], index=dates, name="strategy_A")
#     eq_df = pd.DataFrame({
#         "strategy_A": [100, 102, 101, 99, 98, 97, 99, 100, 101, 103],
#         "strategy_B": [50, 51, 50.5, 52, 51, 50, 49, 51, 52, 53]
#     }, index=dates)
#
#     print("\nFrom Series Sharpe:", sharpe_ratio(eq_series))
#     print("From DF Sharpe:", sharpe_ratio(eq_df))
