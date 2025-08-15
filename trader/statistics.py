import pandas as pd
import numpy as np

DECIMALS = 4
YEAR = 252


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
        return data
    elif isinstance(data, pd.DataFrame):
        return data.squeeze()
    else:
        raise TypeError(f"Input must be a pandas DataFrame or Series, got {type(data)}")


def daily_returns(equity) -> pd.DataFrame:
    """Compute daily returns from equity curve."""
    equity = _ensure_series(equity)
    return round_float(equity.dropna().pct_change().fillna(0))


def annual_returns(equity) -> float:
    daily = daily_returns(equity)
    annual = (1 + daily.mean()) ** YEAR - 1

    return round_float(annual)


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


def max_drawdown_and_duration(equity) -> dict:
    """Calculate max drawdown and its duration for each column."""
    equity = _ensure_series(equity)
    running_max = equity.cummax()
    drawdown = (equity / running_max) - 1.0
    max_dd = drawdown.min()
    duration = equity.index.to_series().groupby((equity == running_max).cumsum()).cumcount()
    max_duration = duration.max()

    return round_float(max_dd), max_duration

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
