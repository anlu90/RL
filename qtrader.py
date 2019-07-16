import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from collections import defaultdict

class QTrader(object):
    def __init__(self):
        self.df1 = pd.read_csv('./GSPC.csv', index_col='Date')
        self.df1.index = pd.to_datetime(self.df1.index)
        self.df2 = pd.read_csv('./tbill.csv', index_col='Date')
        self.df2.index = pd.to_datetime(self.df2.index)
        self.stock_data = pd.merge(self.df1, self.df2, right_index=True, left_index=True).sort_index()
        self.returns = pd.DataFrame({'stocks': self.stock_data['Adj Close'].rolling(window=2).apply(lambda x: x[1]/x[0]-1, raw=False),
                                     'tbills': (self.stock_data['tbill_rate']/100+1)**(1/52)-1}, index=self.stock_data.index)
        self.returns['risk_adjusted'] = self.returns.stocks - self.returns.tbills
    def buy_and_hold(self, dates):
        return pd.Series(1, index = dates)

    def buy_tbills(self, dates):
        return pd.Series(0, index = dates)

    def random(self, dates):
        return pd.Series(np.random.randint(-1,2, size=len(dates)), index = dates)

    def evaluate(self, holdings):
        return pd.Series(self.returns.tbills + holdings * (self.returns.risk_adjusted) +1,
                         index=holdings.index).cumprod()

    def graph_portfolio(self):
        midpoint = int(len(self.returns.index)/2)
        training_indexes = self.returns.index[:midpoint]
        testing_indexes = self.returns.index[midpoint:]
        portfolios = pd.DataFrame({
            'buy_and_hold': self.evaluate(self.buy_and_hold(testing_indexes)),
            'buy_tbills': self.evaluate(self.buy_tbills(testing_indexes)),
            'random': self.evaluate(self.random(testing_indexes))
        }, index=testing_indexes)

        portfolios.plot()
        plt.show()

