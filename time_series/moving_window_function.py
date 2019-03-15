#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : moving_window_function.py
# @Author: Shulin Liu
# @Date  : 2019/3/15
# @Desc  : 滑动窗口
import matplotlib.pylab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.Series(np.random.randn(600), index=pd.date_range('7/1/2016', freq='D', periods=600))
print(df.head())
r = df.rolling(window=10)
print(r.mean())
print(r.min())
print(r.max())
print(r.var())
plt.figure(figsize=(15, 5))
df.plot(style='r--')
df.rolling(window=10).mean().plot(style='b')
plt.show()
