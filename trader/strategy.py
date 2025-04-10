#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : strategy.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/3/12 14:26
import pandas as pd
import numpy as np


class Strategy:
    def __init__(self):
        # self.data = data
        pass

    def generate_signal(self, price_data):
        raise NotImplementedError

    def generate(self, a, b):
        # print(a, b)
        if a is None or b is None:
            return 0
        if a > b: return 1

        if a < b: return -1

    def calculate_change_pct(self, price_data):
        """
            涨跌幅 = (当期收盘价-前期收盘价) / 前期收盘价
            :param data: dataframe，带有收盘价
            :return: dataframe，带有涨跌幅
            """
        # data['close_pct'] = (data['close'] - data['close'].shift(1)) \
        #                     / data['close'].shift(1)
        # a = price_data['close'].pct_change()
        # print(len(a))
        # print(len(price_data))
        return price_data['close'].pct_change()

    def calculate_prof_pct(self, data):
        """
        计算单次收益率：开仓、平仓（开仓的全部股数）
        :param data:
        :return:
        """
        # 筛选信号不为0的，并且计算涨跌幅
        data.loc[data['signal'] != 0, 'profit_pct'] = data['close'].pct_change()
        data = data[data['signal'] == -1]  # 筛选平仓后的数据：单次收益
        return data

    def calculate_cum_prof(self, data):
        """
        计算累计收益率（个股收益率）
        :param data: dataframe
        :return:

        理财产品（本金100元）

第1天：3% ：（1+3%） ✖ 100 = 103
第2天：2% ：（1+2%）✖ 以上 = 103 +2.06
第3天：5% : （1+5%）✖ 以上 = 收益 ✖ 以上
第4天：6% ：（1+6%）✖ 以上 = 收益 ✖ 以上
第n天：累计收益 = (1+当天收益率)的累计乘积-1
这里的计算公式为什么需要减1呢？ 因为我们上面的公式都是包括本金的，比如说103应该减去100，只有3元才是我们的利润，所以这里需要减去1，将本金去掉。
        """
        # 累计收益
        # data['cum_profit'] =
        return pd.DataFrame(1 + data['simple_return']).cumprod() - 1

    def caculate_max_drawdown(self, data, window=252):
        """
        计算最大回撤比
        :param data:
        :param window: int, 时间窗口设置，默认为252（日k）
        :return:

        股票最大回撤是指股票或投资组合在特定时期内，从最高点跌至最低点的最大幅度。

这个指标反映了投资的潜在损失，是评估投资风险的一个重要方式。例如，如果某股票的价格在一段时间内从最高点100元降至最低点80元，那么其最大回撤为20%。

视频作者中采用了(谷值 — 峰值)，我感觉是不是应该反过来计算。
        """
        # 选取时间周期中的最大净值
        data['roll_max'] = data['close'].rolling(window=window, min_periods=1).max()

        # 计算当天的回撤比 = (谷值 — 峰值)/峰值 = 谷值/峰值 - 1
        data['daily_dd'] = data['close'] / data['roll_max'] - 1

        # 选取时间周期内最大的回撤比，即最大回撤

        return data['daily_dd'].rolling(window, min_periods=1).min()


class MovingAverageStrategy(Strategy):
    short = 20
    long = 50

    def __init__(self):
        pass

    def generate_signal(self, price_data, remove_na=True):
        price_data.loc[:, 'short'] = price_data["close"].rolling(self.short).mean()
        price_data.loc[:, 'long'] = price_data["close"].rolling(self.long).mean()

        price_data.loc[:, 'signal'] = price_data.apply(lambda row: self.generate(row['close'], row['short']),
                                                       axis=1)
        # return 1 if short_ma.iloc[-1] > long_ma.iloc[-1] else -1  # Buy/Sell

        price_data['simple_return'] = self.calculate_change_pct(price_data)
        price_data['cum_prof'] = self.calculate_cum_prof(price_data)
        price_data['max_drawdown'] = self.caculate_max_drawdown(price_data)

        if remove_na:
            return price_data[price_data['signal'].notna()]

        return price_data


from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class MLStrategy(Strategy):
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

        super.__init__()

    def train(self, features: pd.DataFrame):
        features = features.copy()
        features["target"] = (features["return_1d"].shift(-1) > 0).astype(int)
        features = features.dropna()
        X = features[["sma_ratio"]]
        y = features["target"]
        self.model.fit(X, y)

    def predict_signal(self, features: pd.DataFrame) -> int:
        """
        Predicts whether to buy (1), sell (-1), or hold (0).
        """
        latest = features.tail(1)[["sma_ratio"]]
        pred = self.model.predict_proba(latest)[0][1]  # Probability stock goes up
        return self._action(pred)

    def _action(self, prob_up):
        """threshold-based decision logic"""
        if prob_up > 0.6:
            return 1  # Buy
        elif prob_up < 0.4:
            return -1  # Sell
        else:
            return 0  # Do nothing (neutral) Hold
