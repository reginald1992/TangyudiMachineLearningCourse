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

# 导入数据，并变化标签
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
# 统计各个特征分布
plt.figure(figsize=(8, 6))
for cnt in range(4):
    plt.subplot(2, 2, cnt + 1)
    for lab in ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'):
        plt.hist(x[y == lab, cnt], label=lab, bins=10, alpha=0.3, )
    plt.xlabel(feature_dict[cnt])
    plt.legend(loc='upper right', fancybox=True, fontsize=8)
plt.tight_layout()
plt.show()
# 归一化特征
x_std = StandardScaler().fit_transform(x)
# 手动计算协方差矩阵
mean_vec = np.mean(x_std, axis=0)
cov_mat = (x_std - mean_vec).T.dot((x_std - mean_vec) / (x_std.shape[0] - 1))
print('Covariance matrix \n%s' % cov_mat)
# 使用numpy计算协方差矩阵
print('NumPy covariance matrix: \n%s' % np.cov(x_std.T))
# 计算特征值和特征向量
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' % eig_vecs)
print('Eigenvalues \n%s' % eig_vals)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
print(eig_pairs)
print('----------')
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
# Visually confirm that the list is correctly sorted by decreasing eigenvalues
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])
# 计算信息量:特征值归一化后计算占sum的百分比
tot = sum(eig_vals)
var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
print(var_exp)
cum_var_exp = np.cumsum(var_exp)
print(cum_var_exp)
a = np.array([1, 2, 3, 4])
print(a)
print('-----------')
print(np.cumsum(a))
# 画出信息量变化图
plt.figure(figsize=(6, 4))
plt.bar(range(4), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(4), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
# 提取前2维的特征值和特征向量
matrix_w = np.hstack((eig_pairs[0][1].reshape(4, 1),
                      eig_pairs[1][1].reshape(4, 1)))
print('Matrix W:\n', matrix_w)
# 矩阵相乘得到降维后的特征
new_x = x_std.dot(matrix_w)
# 画图比较降维前后的特征分布
# 降维前
plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), ('blue', 'red', 'green')):
    plt.scatter(x[y == lab, 0], x[y == lab, 1], label=lab, c=col)
plt.xlabel('sepal_len')
plt.ylabel('sepal_wid')
plt.legend(loc='best')
plt.tight_layout()
plt.show()
# 降维后
plt.figure(figsize=(6, 4))
for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), ('blue', 'red', 'green')):
    plt.scatter(new_x[y == lab, 0], new_x[y == lab, 1], label=lab, c=col)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='lower center')
plt.tight_layout()
plt.show()
