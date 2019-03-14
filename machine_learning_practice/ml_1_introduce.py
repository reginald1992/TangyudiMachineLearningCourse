#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ml_1_introduce.py
# @Author: Shulin Liu
# @Date  : 2019/3/14
# @Desc  : 汽车油耗预测
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name"]
cars = pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)
print(cars.head(5))
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)
cars.plot("weight", "mpg", kind='scatter', ax=ax1)
cars.plot("acceleration", "mpg", kind='scatter', ax=ax2)
plt.show()
lr = LinearRegression(fit_intercept=True)
lr.fit(cars[["weight"]], cars["mpg"])
predictions = lr.predict(cars[["weight"]])
print(predictions[0:5])
print(cars["mpg"][0:5])
plt.scatter(cars['weight'], cars['mpg'], c='red')
plt.scatter(cars['weight'], predictions, c='blue')
plt.show()
mse = mean_squared_error(cars["mpg"], predictions)
print(mse)
rmse = mse ** 0.5
print(rmse)
