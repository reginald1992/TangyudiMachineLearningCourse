#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_predict_stock.py
# @Author: Shulin Liu
# @Date  : 2019/3/17
# @Desc  :使用ARIMA预测股票
import pandas as pd
import datetime
import matplotlib.pylab as plt
import seaborn as sns
from matplotlib.pylab import style
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

stockFile = 'T10yr.csv'
stock = pd.read_csv(stockFile, index_col=0, parse_dates=[0])
print(stock.head(10))

stock_week = stock['Close'].resample('W-MON').mean()
stock_train = stock_week['2000':'2015']
stock_train.plot(figsize=(12, 8))
plt.legend(bbox_to_anchor=(1.25, 0.5))
plt.title("Stock Close")
sns.despine()
plt.show()
# 看差分的数据
stock_diff = stock_train.diff()
stock_diff = stock_diff.dropna()
plt.figure()
plt.plot(stock_diff)
plt.title('一阶差分')
plt.show()
