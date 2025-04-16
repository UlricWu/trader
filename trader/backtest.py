#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : backtest.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/4/10 11:22
from __future__ import annotations

from typing import List

from datetime import datetime

# backtest_engine.py
from typing import List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from .portfolio import Fill, Portfolio  # Assuming Fill is a class that handles transactions

from typing import Dict, List


class Strategy:

    def __init__(self, window_size: int = 100):
        self.features = ['return', 'moving_avg', 'volatility']
        self.target = ['return']
        self.window_size = window_size
        self.signals = {}

        self.models = {}

        self.scalers: Dict[str, StandardScaler] = {}

    def generate_features(self, df) -> pd.DataFrame:
        """
        Generate features for the ML model.
        """
        print(f"generate_features: {len(df)}")
        df['return'] = df['close'].pct_change().fillna(0)
        df['moving_avg'] = df['close'].rolling(window=20).mean().fillna(method="bfill")
        df['volatility'] = df['return'].rolling(window=20).std().fillna(method="bfill")

        df["target"] = df["close"].shift(-1) / df["close"] - 1
        # group = group.dropna(subset=["target"])

        # df["sma_5"] = df["close"].rolling(5).mean().fillna(method="bfill")
        # df["sma_10"] = df["close"].rolling(10).mean().fillna(method="bfill")
        #
        # df["feature1"] = df["return"]
        # df["feature2"] = df["sma_5"] / df["sma_10"] - 1
        # df["feature3"] = df["close"].diff().fillna(0)

        return df

        # return df.dropna(subset=["target"])

    def train(self, df: pd.DataFrame, symbol: str) -> None:
        df = self.generate_features(df)
        # print(df)
        X = df[self.features]
        # y = df[self.target].values.ravel()

        y = df["target"].apply(lambda x: 1 if x > 0.01 else (-1 if x < -0.01 else 0))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        self.models[symbol] = model
        self.scalers[symbol] = scaler

    def predict(self, row: pd.DataFrame, symbol: str) -> int:
        # print(row)
        X = row[self.features]
        X_scaled = self.scalers[symbol].transform(X)
        return self.models[symbol].predict(X_scaled)[0]

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0

        for symbol, group in df.groupby("symbol"):
            signals = [0] * self.window_size
            group = group.sort_values("date").reset_index(drop=True)

            for i in range(self.window_size, len(group)):
                window_raw = group.iloc[0:i + 1].copy()  # include i-th row
                window_engineered = self.generate_features(window_raw.iloc[:-1])  # drop last

                self.train(window_engineered, symbol)
                # signal = self.predict(group.iloc[[i]], symbol)

                predict_engineered = self.generate_features(window_raw)  # drop last

                predicted_return = self.predict(predict_engineered.iloc[[i]], symbol)
                # Predict signal (1 = Buy, 0 = Hold, -1 = Sell)
                signal = 1 if predicted_return > 0.01 else (-1 if predicted_return < -0.01 else 0)
                signals.append(signal)

            df.loc[df["symbol"] == symbol, "signal"] = signals

        return df

    @staticmethod
    def predict_signal(prob_up):
        """Return the prediction (buy, sell, hold) for a given symbol."""
        # pass
        if prob_up > 0.6:
            return 1  # Buy
        elif prob_up < 0.4:
            return -1  # Sell
        else:
            return 0  # Do nothing (neutral) Hold


class BacktestEngine:
    def __init__(self, df: pd.DataFrame, strategy, symbols: List[str], initial_cash: float = 10000,
                 slippage: float = 0.001,
                 commission: float = 0.001):
        self.df = df.sort_values(by='date').copy()
        self.strategy = strategy
        self.symbols = symbols
        self.portfolio = Portfolio(initial_cash=initial_cash)

        self.initial_cash = initial_cash

        self.slippage = slippage
        self.commission = commission

    def run(self):
        """Run the backtest by iterating through each trading day."""
        # Create a dictionary to hold portfolio values for each symbol
        # portfolio_values = {symbol: self.initial_cash for symbol in self.symbols}

        # for i in range(self.window_size, len(self.df)):
        # for date in self.df['date'].unique().sort().tolist():

        prices = {}
        self.df = self.strategy.generate_signals(self.df)
        for symbol, group in self.df.groupby("symbol"):
            group = group.sort_values("date").reset_index(drop=True)

            for _, row in group.iterrows():
                signal = row["signal"]
                price = row["close"]
                date = pd.to_datetime(row["date"])

                if signal == 0:
                    continue

                quantity = signal * 100
                fill = Fill(symbol=symbol, quantity=quantity, price=price, date=date)

                # Add slippage as a percentage or fixed value
                #         # execution_price = close * (1 + slippage_pct)
                #         # price += self.commission * price * signal
                #         # size = self.strategy.sizing(history, symbol, equity)
                #
                #         price = self.df.loc[(self.df['date'] == date) & (self.df['symbol'] == symbol)]['close']
                #         size = 100
                #
                #         fills.append(Fill(symbol=symbol, quantity=size, price=price, date=date))
                #         prices[symbol] = price
                self.portfolio.update_position(fill)
                prices[symbol] = price

                self.portfolio.mark_to_market(prices, date)

    def current_value(self) -> float:
        total = self.portfolio.cash
        for symbol, pos in self.portfolio.positions.items():
            latest_price = self.df[self.df["symbol"] == symbol]["close"].iloc[-1]
            total += pos.quantity * latest_price
        return total

    def get_results(self) -> dict:
        return {
            "final_value": round(self.current_value(), 2),
            "cash": round(self.portfolio.cash, 2),
            "positions": self.portfolio.summary()
        }
