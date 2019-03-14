#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ml_2_logistic_regression.py
# @Author: Shulin Liu
# @Date  : 2019/3/14
# @Desc  : 逻辑回归判断是否录取
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

admissions = pd.read_csv("admissions.csv")
print(admissions.head())
plt.scatter(admissions['gpa'], admissions['admit'])
plt.show()


def logit(x):
    """
    计算概率函数
    :param x:原始数据
    :return:概率
    """
    return np.exp(x) / (1 + np.exp(x))


# Generate 50 real values, evenly spaced, between -6 and 6.
x = np.linspace(-6, 6, 50, dtype=float)
# Transform each number in t using the logit function.
y = logit(x)
# Plot the resulting data.
plt.plot(x, y)
plt.ylabel("Probability")
plt.show()
# 使用线性回归计算
linear_model = LinearRegression()
linear_model.fit(admissions[["gpa"]], admissions["admit"])
# 使用逻辑回归计算
logistic_model = LogisticRegression()
logistic_model.fit(admissions[["gpa"]], admissions["admit"])
# 预测概率
pred_probs = logistic_model.predict_proba(admissions[["gpa"]])
plt.scatter(admissions["gpa"], pred_probs[:, 1])
plt.show()
# 预测分类结果
fitted_labels = logistic_model.predict(admissions[["gpa"]])
plt.scatter(admissions["gpa"], fitted_labels)
plt.show()
