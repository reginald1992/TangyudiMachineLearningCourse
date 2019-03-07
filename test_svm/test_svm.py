#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_svm.py
# @Author: Shulin Liu
# @Date  : 2019/3/7
# @Desc  : 支持向量机调参实例
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.datasets.samples_generator import make_blobs, make_circles
from sklearn.svm import SVC  # support vector classifier

# 生成数据
x, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap="autumn")
plt.show()
# 画图
xfit = np.linspace(-1, 3.5)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
plt.plot([0.6], [2.1], 'x', color='red', markeredgewidth=2, markersize=10)
for m, b in [(1, 0.65), (0.5, 1.6), (-0.2, 2.9)]:
    plt.plot(xfit, m * xfit + b, '-k')
plt.xlim(-1, 3.5)
plt.show()
# SVM 最小化 雷区
xfit = np.linspace(-1, 3.5)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)
plt.xlim(-1, 3.5)
plt.show()
# 训练一个基本的svm
model = SVC(kernel='linear')
model.fit(x, y)


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """
     Plot the decision function for a 2D SVC
    :param model:模型
    :param ax:
    :param plot_support:
    :return:
    """
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.show()
"""
这条线就是我们希望得到的决策边界啦
观察发现有3个点做了特殊的标记，它们恰好都是边界上的点
它们就是我们的support vectors（支持向量）
在Scikit-Learn中, 它们存储在这个位置 support_vectors_（一个属性）
观察可以发现，只需要支持向量我们就可以把模型构建出来
接下来我们尝试一下，用不同多的数据点，看看效果会不会发生变化
分别使用60个和120个数据点
"""


def plot_svm(N=10, ax=None):
    """
    比较不同个数的点对效果的影响
    :param N:
    :param ax:
    :return:
    """
    x, y = make_blobs(n_samples=200, centers=2,
                      random_state=0, cluster_std=0.60)
    x = x[:N]
    y = y[:N]
    model = SVC(kernel='linear', C=1E10)
    model.fit(x, y)

    ax = ax or plt.gca()
    ax.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)


fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title('N = {0}'.format(N))
plt.show()

# 引入核函数的支持向量机
x, y = make_circles(n_samples=100, factor=0.1, noise=0.1)
clf = SVC(kernel='linear').fit(x, y)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf, plot_support=False)
plt.show()

#加入了新的维度r
from mpl_toolkits import mplot3d
r = np.exp(-(x ** 2).sum(1))


def plot_3d(elev=30, azim=30, x=x, y=y):
    ax = plt.subplot(projection='3d')
    ax.scatter3D(x[:, 0], x[:, 1], r, c=y, s=50, cmap='autumn')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('r')


plot_3d(elev=45, azim=45, x=x, y=y)
plt.show()
# 加入径向基函数
clf = SVC(kernel='rbf', C=1E6)
clf.fit(x, y)
# 画图展示分类效果
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=300, linewidths=1)
plt.show()
'''
调节SVM参数：soft margin问题
调节C参数：
C参数趋近于无穷大时：意味着分类严格不能有错误
C参数趋近于很小时：意味着分类可以容忍更多的错误
'''
x, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)
plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

for axi, C in zip(ax, [10.0, 0.1]):
    model = SVC(kernel='linear', C=C).fit(x, y)
    axi.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(model, axi)
    axi.scatter(model.support_vectors_[:, 0],
                model.support_vectors_[:, 1],
                s=300, lw=1, facecolors='none')
    axi.set_title('C = {0:.1f}'.format(C), size=14)
plt.show()