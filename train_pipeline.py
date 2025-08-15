# train_pipeline.py
import os

import numpy as np
from trader.backtest_engine import Backtest

from trader.config import load_settings

from trader.performance import PerformanceAnalyzer

from trader.strategy import MLStrategy


def train():
    settings = load_settings()

    """Train pipeline in batch mode."""
    print("[Train Pipeline Started]")

    from data import db

    codes = ["000001.SZ", "000002.SZ"]

    df = db.extract_table(database='db/tutorial.db', end_day="20250205", start_day='20241001', ts_code=codes)
    data = db.load_and_normalize_data(df)
    print(len(data))

    bt = Backtest(data=data, settings=settings, strategy=MLStrategy)
    bt.run()
    print(bt.summary())
    perf = PerformanceAnalyzer(portfolio=bt.portfolio)
    print(perf.summary())
    perf.plot()

    # print(f"🎯 Training for symbol: {symbol}")
    # bars = data_handler.get_symbol_bars(symbol)
    #
    # # Prepare training data from bars
    # # X_train, y_train, X_test, y_test = prepare_training_data(bars, settings)
    # # X, y = data_handler.get_ml_features_and_targets(symbol)
    # print(f'symbol={symbol}, bars={len(bars)}, X_train={X_train.shape}, y_train={y_train.shape}')
    #
    # # Train model
    # model = MLSignalGenerator()
    #
    # strategy = MLStrategy(data_handler, events, symbol="AAPL")
    # portfolio = Portfolio(data_handler, events, initial_cash=100000)
    # execution_handler = SimulatedExecutionHandler(events)
    # performance = PerformanceAnalyzer()
    #
    # bt = Backtest(
    #     data_handler, strategy, portfolio,
    #     execution_handler, performance, events
    # )
    # bt.run()
    # # bt.plot_results()
    # # model = Model(settings=settings, symbol=symbol)
    # # model.train(X_train, y_train)
    #
    # pred = model.predict(X_test)
    #
    # print(f"\n[Train Pipeline Completed ✅] {symbol}")

    print("🏁 All symbol models trained and saved.")


# def prepare_training_data(bars, settings):
#     """Generate features and labels for training."""
#     close_prices = bars['close'].values
#     X_train, y_train, X_test, y_test = [], [], [], []
#
#     for i in range(settings.model.lookback, len(close_prices) - 2):
#         short_ma = np.mean(close_prices[i - settings.strategy.short_window:i])
#         long_ma = np.mean(close_prices[i - settings.strategy.long_window:i])
#
#         feature = [short_ma, long_ma]
#         label = 1 if close_prices[i + 1] > close_prices[i] else 0
#
#         X_train.append(feature)
#         y_train.append(label)
#
#     i+=1
#     short_ma = np.mean(close_prices[i - settings.strategy.short_window:i])
#     long_ma = np.mean(close_prices[i - settings.strategy.long_window:i])
#
#     feature = [short_ma, long_ma]
#     label = 1 if close_prices[i + 1] > close_prices[i] else 0
#
#
#     return np.array(X_train), np.array(y_train), np.array(feature), np.array(label)


if __name__ == "__main__":
    # your injected settings

    train()
