#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : time_series.py
# @Author: Shulin Liu
# @Date  : 2019/3/15
# @Desc  : 时间序列分析
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
时间序列
时间戳（timestamp）
固定周期（period）
时间间隔（interval）
"""
'''
date_range
可以指定开始时间与周期
H：小时
D：天
M：月
'''
# TIMES #2016 Jul 1 7/1/2016 1/7/2016 2016-07-01 2016/07/01
rng = pd.date_range('7/1/2016', periods=10, freq='D')
print(rng)
time = pd.Series(np.random.randn(20), index=pd.date_range(datetime.datetime(2016, 1, 1), periods=20))
print(time)
print(time['2016-01-15'])
data = pd.date_range('2010-01-01', '2011-01-01', freq='M')
print(data)
# 过滤truncate
print(time.truncate(before='2016-1-15'))
print(time.truncate(after='2016-1-7'))
# 时间戳
print(pd.Timestamp('2016-07-10'))
# 可以指定更多细节
print(pd.Timestamp('2016-07-10 10'))
print(pd.Timestamp('2016-07-10 10:15'))
# 时间区间
print(pd.Period('2016-01'))
print(pd.Period('2016-01-01'))
# TIME OFFSETS
print(pd.Timedelta('1 day'))
print(pd.Period('2016-01-01 10:10') + pd.Timedelta('1 day'))
print(pd.Timestamp('2016-01-01 10:10') + pd.Timedelta('1 D'))
p1 = pd.period_range('2016-01-01 10:10', freq='25H', periods=10)
p2 = pd.period_range('2016-01-01 11:15', freq='1D1H', periods=10)
print(p1)
print(p2)
# 指定索引
rng = pd.date_range('2016 Jul 1', periods=10, freq='D')
print(pd.Series(range(len(rng)), index=rng))
periods = [pd.Period('2016-01'), pd.Period('2016-02'), pd.Period('2016-03')]
ts = pd.Series(np.random.randn(len(periods)), index=periods)
print(ts)
