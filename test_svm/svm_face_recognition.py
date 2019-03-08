#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : svm_face_recognition.py
# @Author: Shulin Liu
# @Date  : 2019/3/8
# @Desc  :
"""
As an example of support vector machines in action,
let's take a look at the facial recognition problem.
We will use the Labeled Faces in the Wild dataset,
which consists of several thousand collated photos of various public figures.
A fetcher for the dataset is built into Scikit-Learn:
"""
from sklearn.datasets import fetch_lfw_people
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)
# Let's plot a few of these faces to see what we're working with:
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
plt.show()
# 每个图的大小是 [62×47]
# 在这里把每一个像素点当成了一个特征，但是这样特征太多了，用PCA降维一下吧！
pca = PCA(n_components=150, whiten=True, random_state=0)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)
x_train, x_test, y_train, y_test = train_test_split(faces.data, faces.target, random_state=0)
# 使用grid search cross_validation 来选择参数
param = {'svc__C': [1, 5, 10], 'svc__gamma': [0.0001, 0.0005, 0.001]}
grid = GridSearchCV(model, param)
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
model = grid.best_estimator_
y_fit = model.predict(x_test)
# 画出结果
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(x_test[i].reshape(62, 47), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[y_fit[i]].split()[-1],
                   color='black' if y_fit[i] == y_test[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
plt.show()

print(classification_report(y_test, y_fit, target_names=faces.target_names))
"""
精度(precision) = 正确预测的个数(TP)/被预测正确的个数(TP+FP)
召回率(recall)=正确预测的个数(TP)/预测个数(TP+FN)
F1 = 2精度召回率/(精度+召回率)
"""
mat = confusion_matrix(y_test, y_fit)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
