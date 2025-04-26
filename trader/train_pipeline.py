# train_pipeline.py
from trader import get_strategy

from queue import Queue
from trader.data_handler import DailyBarDataHandler
from trader.events import EventType


# class LivePipeline:
#     def __init__(self, data):
#         self.events = Queue()
#         self.data_handler = DailyBarDataHandler(data=data, events=self.events)
#         self.strategy = get_strategy(self.events)
#
#     def run_live(self):
#         # Live trading
#         ...

# train_pipeline.py
class TrainPipeline:
    def __init__(self, data):
        self.events = Queue()
        self.data_handler = DailyBarDataHandler(data=data, events=self.events)
        self.strategy = get_strategy(self.events)

    def run_training(self):
        while self.data_handler.continue_backtest:
            self.data_handler.stream_next()
            while not self.events.empty():
                event = self.events.get()
                if event.type == EventType.MARKET:
                    self.strategy.on_market(event)

        self.strategy.train_model()


# train_pipeline.py
from trader.config import load_settings
from trader.strategy import MLStrategy

def train():

    from data import db

    code = "000001.SZ"

    df = db.extract_table(day="20250205", start_day='20240601', ts_code=[code])
    data = db.load_and_normalize_data(df)

    # _get_adjustment

    # Load data
    # Load data
    # data = load_data(
    #     adjustment=settings.data.price_adjustment,
    #     symbols=settings.trading.symbol_list
    # )

    # Initialize and train model
    strategy = MLStrategy(settings)
    strategy.train(data)

    # Save if needed
    if settings.model.auto_save:
        strategy.save_model()




if __name__ == "__main__":
    settings = load_settings()
    train(settings)

