#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ml_4_cross_validation.py
# @Author: Shulin Liu
# @Date  : 2019/3/15
# @Desc  : 交叉验证
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

admissions = pd.read_csv("admissions.csv")
admissions["actual_label"] = admissions["admit"]
admissions = admissions.drop("admit", axis=1)
print(admissions.head())

np.random.seed(8)
shuffled_index = np.random.permutation(admissions.index)
# print shuffled_index
shuffled_admissions = admissions.loc[shuffled_index]
train = shuffled_admissions.iloc[0:515]
test = shuffled_admissions.iloc[515:len(shuffled_admissions)]
print(shuffled_admissions.head())

model = LogisticRegression()
model.fit(train[["gpa"]], train["actual_label"])
labels = model.predict(test[["gpa"]])
test["predicted_label"] = labels
matches = test["predicted_label"] == test["actual_label"]
correct_predictions = test[matches]
accuracy = len(correct_predictions) / float(len(test))
print('accuracy: %f' % accuracy)
true_positive_filter = (test["predicted_label"] == 1) & (test["actual_label"] == 1)
true_positives = len(test[true_positive_filter])
false_negative_filter = (test["predicted_label"] == 0) & (test["actual_label"] == 1)
false_negatives = len(test[false_negative_filter])

sensitivity = true_positives / float((true_positives + false_negatives))
print(sensitivity)

false_positive_filter = (test["predicted_label"] == 1) & (test["actual_label"] == 0)
false_positives = len(test[false_positive_filter])
true_negative_filter = (test["predicted_label"] == 0) & (test["actual_label"] == 0)
true_negatives = len(test[true_negative_filter])

specificity = (true_negatives) / float((false_positives + true_negatives))
print(specificity)

probabilities = model.predict_proba(test[["gpa"]])
fpr, tpr, thresholds = metrics.roc_curve(test["actual_label"], probabilities[:, 1])
print(thresholds)
plt.plot(fpr, tpr)
plt.show()
# Means we can just use roc_auc_curve() instead of metrics.roc_auc_curve()
auc_score = roc_auc_score(test["actual_label"], probabilities[:,1])
print('auc score: %f' % auc_score)