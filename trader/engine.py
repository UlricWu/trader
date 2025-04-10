#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : engine.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/3/12 14:07
import queue


class BaseEngine(object):
    def __init__(self):
        pass

    def run(self, symbol):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class EventEngine(BaseEngine):
    def __init__(self):
        self.events = queue.Queue()

    def add_event(self, event):
        self.events.put(event)

    def run(self):
        while not self.events.empty():
            event = self.events.get()
            # Process event


class StrategyEngine(BaseEngine):
    def __init__(self):
        pass

    def run(self):
        pass

    def buy(self):
        raise NotImplementedError

    def sell(self):
        raise NotImplementedError


class RsiStrategyEngine(StrategyEngine):
    rsi_window = 14
    rsi_min = 0
    rsi_max = 10

    rsi_step = 1

    def __init__(self):
        super().__init__()
        self.rsi = 20

    def run(self):
        if self.rsi > self.rsi_max:
            self.position = False  # close

        if self.rsi < self.rsi_min:
            self.position = True  # buy


class DemoStrategyEngine(StrategyEngine):
    def __init__(self):
        pass

    def run(self):
        # if current_price > avg(10):
        #     self.position = True
        # elif current_price < avg(10):
        #     self.position = False
        # 如果价格高于最近10天的平均价格就买入，如果价格低于最近10天的平均价格就卖出
        pass


class BacktestEngine(BaseEngine):
    def __init__(self, bars, start_date=None, end_date=None):
        super(BacktestEngine, self).__init__()
        self.events = queue.Queue()
        self.bars = bars

    def run(self):
        while self.bars.update_bars():
            # 可能有多事件，要处理完，如果队列暂时是空的，不处理
            while True:
                # 队列为空则消息循环结束
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    break
                else:
                    self.handle_event(event)

    def handle_event(self, event):
        pass
