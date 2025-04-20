# data_handler.py
import pandas as pd
from events import MarketEvent
import datetime


class DailyBarDataHandler:
    mock_data = pd.DataFrame([
        {"datetime": datetime.date(2023, 1, 1), "open": 100, "high": 101, "low": 99, "close": 100.5},
        {"datetime": datetime.date(2023, 1, 2), "open": 100.5, "high": 102, "low": 100, "close": 101},
        {"datetime": datetime.date(2023, 1, 3), "open": 101, "high": 103, "low": 100, "close": 102},
    ])

    def __init__(self, events, data: pd.DataFrame, symbol: str):
        self.events = events
        self.data = data if len(data) > 0 else self.mock_data
        self.symbol = symbol
        self.current_idx = 0
        self.continue_backtest = True

    def update_market(self):
        if self.current_idx >= len(self.data):
            self.continue_backtest = False
            return

        row = self.data.iloc[self.current_idx]
        event = MarketEvent(
            datetime=row["date"],
            symbol=self.symbol,
            price=row["close"],
        )
        self.events.put(event)
        self.current_idx += 1
