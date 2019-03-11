#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_kmeans.py
# @Author: Shulin Liu
# @Date  : 2019/3/11
# @Desc  : 使用K-menas压缩图片
from skimage import io
from sklearn.cluster import KMeans
import numpy as np

image = io.imread('test2.jpg')
io.imshow(image)
io.show()

rows = image.shape[0]
cols = image.shape[1]
image = image.reshape(image.shape[0] * image.shape[1], 3)
kmeans = KMeans(n_clusters=128, n_init=10, max_iter=200)
kmeans.fit(image)
clusters = np.asarray(kmeans.cluster_centers_, dtype=np.uint8)
labels = np.asarray(kmeans.labels_, dtype=np.uint8)
labels = labels.reshape(rows, cols)
print(clusters.shape)
np.save('codebook_test.npy', clusters)
io.imsave('compressed_test.jpg', labels)
