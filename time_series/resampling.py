#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : resampling.py
# @Author: Shulin Liu
# @Date  : 2019/3/15
# @Desc  :时间重采样
import pandas as pd
import numpy as np
"""
数据重采样
时间数据由一个频率转换到另一个频率
降采样
升采样
"""
rng = pd.date_range('1/1/2011', periods=90, freq='D')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts.head())
print(ts.resample('M').sum())
print(ts.resample('3D').sum())
day3Ts = ts.resample('3D').mean()
print(day3Ts)
print(day3Ts.resample('D').asfreq())
"""
插值方法：
ffill 空值取前面的值
bfill 空值取后面的值
interpolate 线性取值
"""
print(day3Ts.resample('D').ffill(1))
print(day3Ts.resample('D').bfill(1))
print(day3Ts.resample('D').interpolate('linear'))
