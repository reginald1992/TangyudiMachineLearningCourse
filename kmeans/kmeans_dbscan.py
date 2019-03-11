#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : kmeans_dbscan.py
# @Author: Shulin Liu
# @Date  : 2019/3/11
# @Desc  : 聚类算法实例，使用kmeans和dbscan
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.cluster import KMeans, DBSCAN
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
# 读取数据并聚类
beer = pd.read_csv('data.txt', sep=' ')
x = beer[['calories', 'sodium', 'alcohol', 'cost']]
km = KMeans(n_clusters=3).fit(x)
km2 = KMeans(n_clusters=2).fit(x)
print(km.labels_)
print(km2.labels_)
beer['cluster'] = km.labels_
beer['cluster2'] = km2.labels_
print(beer.sort_values('cluster'))
# 聚类中心点
cluster_centers = km.cluster_centers_
cluster_centers2 = km2.cluster_centers_
print(beer.groupby('cluster').mean())
print(beer.groupby('cluster2').mean())
centers = beer.groupby('cluster').mean().reset_index()
# 画图
plt.rcParams['font.size'] = 14
colors = np.array(['red', 'green', 'blue', 'yellow'])
plt.scatter(beer["calories"], beer["alcohol"], c=colors[beer["cluster"]])
plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')
plt.xlabel("Calories")
plt.ylabel("Alcohol")
plt.show()
# 多维数据可视化
scatter_matrix(beer[["calories", "sodium", "alcohol", "cost"]],
               s=100, alpha=1, c=colors[beer["cluster"]], figsize=(10, 10))
plt.suptitle("With 3 centroids initialized")
plt.show()
# 对数据做归一化
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
km = KMeans(n_clusters=3).fit(x_scaled)
beer['scaled_cluster'] = km.labels_
print(beer.sort_values('scaled_cluster'))
print(beer.groupby('scaled_cluster').mean())
scatter_matrix(x, c=colors[beer.scaled_cluster], alpha=1, figsize=(10, 10), s=100)
plt.show()
# 聚类评价指标：轮廓系数
score_scaled = metrics.silhouette_score(x, beer.scaled_cluster)
score = metrics.silhouette_score(x, beer.cluster)
print(score_scaled, score)
scores = []
for k in range(2, 20):
    labels = KMeans(n_clusters=k).fit(x).labels_
    score = metrics.silhouette_score(x, labels)
    scores.append(score)
print(scores)
plt.plot(list(range(2, 20)), scores)
plt.xlabel('Number of Cluster Initialized')
plt.ylabel('Silhouette Score')
plt.show()
# 使用DBSCAN做聚类
db = DBSCAN(eps=10, min_samples=2).fit(x)
labels = db.labels_
beer['cluster_db'] = labels
print(beer.sort_values('cluster_db'))
