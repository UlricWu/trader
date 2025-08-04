# data_handler.py
from typing import Optional

import pandas as pd
from trader.events import MarketEvent

from datetime import datetime
from typing import Dict, List
from trader.events import MarketEvent
from utilts.logs import logs
from trader.config import Settings


class DailyBarDataHandler:
    """
    Streams daily bar data (OHLCV) one day at a time for all symbols.
    Emits MarketEvent per symbol per day.
    """

    def __init__(self, data: pd.DataFrame, events, settings: Settings):
        """
        Parameters:
        - data: A pandas DataFrame with columns [date, symbol, open, high, low, close]
        - events: A queue to place MarketEvent objects
        """
        self.symbol_data = data

        self.events = events
        self.continue_backtest = True

        self._index_iter = self._generate_index_iterator()
        self.settings = settings

    @property
    def symbols(self) -> List[str]:
        return list(self.symbol_data['symbol'].unique())

    def get_symbol_bars(self, symbol):
        return self.symbol_data[self.symbol_data['symbol'] == symbol]

    def _generate_index_iterator(self):
        self.symbol_data.sort_values(by=["date", "symbol"], inplace=True)
        index = self.symbol_data['date']
        for date in index:
            yield date

    # def apply_adjustments(self, df: pd.DataFrame) -> pd.DataFrame:
    #     adj_factors = tushare_api.get_adj_factors(...)
    #     df["adj_factor"] = df["date"].map(adj_factors)
    #     df[["open", "high", "low", "close"]] *= df["adj_factor"]
    #     return df

    # def _apply_adjustment(self, data):
    #     # adjust_type = MODE_TO_ADJUST.get(RUN_MODE, "none")
    #     if adjust_type == "qfq":
    #         return self._forward_adjust(data)
    #     elif adjust_type == "hfq":
    #         return self._backward_adjust(data)
    #     else:
    #         return data

    def _forward_adjust(self, data):
        # Implement your forward adjustment logic
        return data

    def _backward_adjust(self, data):
        # Implement your backward adjustment logic
        return data

    def stream_next(self):

        """
        Push all MarketEvent(s) for the current date (for all symbols) into the event queue.
        """
        try:
            current_date = next(self._index_iter)
        except StopIteration:
            self.continue_backtest = False
            return

        bars = self.symbol_data[self.symbol_data["date"] == current_date]
        for _, bar in bars.iterrows():
            # adjusted_bar = self._adjust_price(bar, symbol)
            event = MarketEvent(
                datetime=bar["date"],
                symbol=bar["symbol"],
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],

            )
            if self.events:
                self.events.put(event)
