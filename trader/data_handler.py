# data_handler.py
import pandas as pd
from .events import MarketEvent
import datetime


class DailyBarDataHandler:

    def __init__(self, events, data: pd.DataFrame):
        self.events = events
        self.data = data

        self.data.sort_values(["date", "symbol"], inplace=True)

        self.symbols = self.data["symbol"].unique().tolist()
        self.dates = self.data["date"].unique().tolist()

        self.current_idx = 0

    @property
    def continue_backtest(self):
        return self.current_idx < len(self.dates)

    def stream_next(self):
        if not self.continue_backtest: return

        current_date = self.dates[self.current_idx]
        daily_data = self.data[self.data["date"] == current_date]
        for _, row in daily_data.iterrows():
            event = MarketEvent(
                datetime=row["date"],
                symbol=row["symbol"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"]
            )
            self.events.put(event)

        self.current_idx += 1  # only track the days
