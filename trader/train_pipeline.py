# train_pipeline.py
import os

import numpy as np

from queue import Queue
from trader.data_handler import DailyBarDataHandler
from trader.events import EventType
from trader.model import Model

# train_pipeline.py


# train_pipeline.py
from trader.config import load_settings


def train(settings):
    """Train pipeline in batch mode."""
    print("[Train Pipeline Started]")

    from data import db

    code = "000001.SZ"

    df = db.extract_table(day="20250205", start_day='20240601', ts_code=[code])
    data = db.load_and_normalize_data(df)

    # _get_adjustment

    # Load data
    # Data Handler - already forward-adjusted (QFQ)
    # data = load_data(
    #     adjustment=settings.data.price_adjustment,
    #     symbols=settings.trading.symbol_list
    # )

    # Initialize and train model
    # 2. Initialize DataHandler
    data_handler = DailyBarDataHandler(data=data,
                                       events=None,  # No events needed in training
                                       settings=settings)

    symbols = data_handler.symbols  # Get all available symbols
    print(f"🧠 Found symbols for training: {symbols}")

    for symbol in symbols:
        print(f"🎯 Training for symbol: {symbol}")
        bars = data_handler.get_symbol_bars(symbol)

        # Prepare training data from bars
        X_train, y_train = prepare_training_data(bars, settings)
        # X, y = data_handler.get_ml_features_and_targets(symbol)

        # Train model
        model = Model(settings=settings, symbol=symbol)
        model.train(X_train, y_train)

        print(f"\n[Train Pipeline Completed ✅] {symbol}")

    print("🏁 All symbol models trained and saved.")


def prepare_training_data(bars, settings):
    """Generate features and labels for training."""
    close_prices = bars['close'].values
    X, y = [], []

    for i in range(settings.model.lookback, len(close_prices) - 1):
        short_ma = np.mean(close_prices[i - settings.strategy.short_window:i])
        long_ma = np.mean(close_prices[i - settings.strategy.long_window:i])

        feature = [short_ma, long_ma]
        label = 1 if close_prices[i + 1] > close_prices[i] else 0

        X.append(feature)
        y.append(label)

    return np.array(X), np.array(y)


if __name__ == "__main__":
    # your injected settings
    settings = load_settings()

    train(settings)
