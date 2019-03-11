#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_pca.py
# @Author: Shulin Liu
# @Date  : 2019/3/11
# @Desc  : PCA实例
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('iris.data')
df.columns = ['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
x = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values
label_dict = {1: 'Iris-Setosa',
              2: 'Iris-Versicolor',
              3: 'Iris-Virgnica'}

feature_dict = {0: 'sepal length [cm]',
                1: 'sepal width [cm]',
                2: 'petal length [cm]',
                3: 'petal width [cm]'}

plt.figure(figsize=(8, 6))
for cnt in range(4):
    plt.subplot(2, 2, cnt+1)
    for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
        plt.hist(x[y == lab, cnt], label=lab, bins=10, alpha=0.3,)
    plt.xlabel(feature_dict[cnt])
    plt.legend(loc='upper right', fancybox=True, fontsize=8)
plt.tight_layout()
plt.show()

x_std = StandardScaler().fit_transform(x)
mean_vec = np.mean(x_std, axis=0)
cov_mat = (x_std - mean_vec).T.dot((x_std - mean_vec) / (x_std.shape[0] - 1))
print('Covariance matrix \n%s' % cov_mat)
