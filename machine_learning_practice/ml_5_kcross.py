#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ml_5_kcross.py
# @Author: Shulin Liu
# @Date  : 2019/3/15
# @Desc  : 交叉验证
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

admissions = pd.read_csv("admissions.csv")
admissions["actual_label"] = admissions["admit"]
admissions = admissions.drop("admit", axis=1)

shuffled_index = np.random.permutation(admissions.index)
shuffled_admissions = admissions.loc[shuffled_index]
admissions = shuffled_admissions.reset_index()
admissions.loc[0:128, "fold"] = 1
admissions.loc[129:257, "fold"] = 2
admissions.loc[258:386, "fold"] = 3
admissions.loc[387:514, "fold"] = 4
admissions.loc[515:644, "fold"] = 5
# Ensure the column is set to integer type.
admissions["fold"] = admissions["fold"].astype('int')

print(admissions.head())
print(admissions.tail())

# Training
model = LogisticRegression()
train_iteration_one = admissions[admissions["fold"] != 1]
test_iteration_one = admissions[admissions["fold"] == 1]
model.fit(train_iteration_one[["gpa"]], train_iteration_one["actual_label"])

# Predicting
labels = model.predict(test_iteration_one[["gpa"]])
test_iteration_one["predicted_label"] = labels

matches = test_iteration_one["predicted_label"] == test_iteration_one["actual_label"]
correct_predictions = test_iteration_one[matches]
iteration_one_accuracy = len(correct_predictions) / float(len(test_iteration_one))
print('iteration_one_accuracy: %f' % iteration_one_accuracy)

fold_ids = [1, 2, 3, 4, 5]


def train_and_test(df, folds):
    fold_accuracies = []
    for fold in folds:
        model = LogisticRegression()
        train = admissions[admissions["fold"] != fold]
        test = admissions[admissions["fold"] == fold]
        model.fit(train[["gpa"]], train["actual_label"])
        labels = model.predict(test[["gpa"]])
        test["predicted_label"] = labels
        matches = test["predicted_label"] == test["actual_label"]
        correct_predictions = test[matches]
        fold_accuracies.append(len(correct_predictions) / float(len(test)))
    return(fold_accuracies)


accuracies = train_and_test(admissions, fold_ids)
print(accuracies)
average_accuracy = np.mean(accuracies)
print('average_accuracy %f' % average_accuracy)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

admissions = pd.read_csv("admissions.csv")
admissions["actual_label"] = admissions["admit"]
admissions = admissions.drop("admit", axis=1)

kf = KFold(n_splits=5, shuffle=True, random_state=6)
lr = LogisticRegression()
# roc_auc
accuracies = cross_val_score(lr, admissions[["gpa"]], admissions["actual_label"], scoring="roc_auc", cv=kf)
average_accuracy = sum(accuracies) / len(accuracies)
print(accuracies)
print(average_accuracy)
