#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @File    : viewer.py
# @Project : trader
# @Author  : wsw
# @Time    : 2025/3/19 11:11


import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns


class Visualizer:
    def __init__(self, df):
        # self.df = df
        sns.set_style("darkgrid")

        df['trade_date'] = pd.to_datetime(df['trade_date'], format="%Y%m%d")

        # nrows, ncols = 2, 1
        # names = ['b', 'c']
        # # Creating subplot axes
        # fig, axes = plt.subplots(nrows, ncols)
        #
        # # Iterating through axes and names
        # for name, ax in zip(names, axes.flatten()):
        #     sns.boxplot(y=name, x="a", data=df, orient='v', ax=ax)

        fig, axarr = plt.subplots(2, figsize=(20, 10), sharex=True)

        sns.lineplot(x="trade_date", y="simple_return", data=df, ax=axarr[0], linewidth=1)

        sns.lineplot(x="trade_date", y="close", data=df, ax=axarr[1],
                     label="Close Price", linewidth=1)

        # # Plot markers for predictions using seaborn scatterplot
        up_df = df[df['signal'] == 1]
        down_df = df[df['signal'] == -1]
        sns.scatterplot(x=up_df['trade_date'], y=up_df['close'], color='red', marker='o', s=10, label='Up', ax=axarr[1])
        sns.scatterplot(x=down_df['trade_date'], y=down_df['close'], color='green', marker='X', s=10, label='Down',
                        ax=axarr[1])

        # set title and remove xticks, labels etc.
        titles = [
            "Portfolio Return",
            "Trading Price Over Time"
        ]
        for i, title in enumerate(titles):
            axarr[i].set(title=title, xticks=[], xticklabels=[], xlabel=None)

        axarr[-1].set(xlabel="Trade Date")
        fig.suptitle("Portfolio Performance")
        fig.show()

    def plot_returns(self):
        pass

    def plot_pnl(self):
        pass

        # Mark predictions
        # for _, row in df.iterrows():
        #     if row['signal'] == 1:
        #         marker, color = 'o', 'red'
        #     else:
        #         marker, color = 'x', 'green'
        #     ax.scatter(row['trade_date'], row['close'], color=color, marker=marker, s=5, label='_nolegend_')

        # Separate up and down predictions
