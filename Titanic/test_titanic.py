#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_titanic.py
# @Author: Shulin Liu
# @Date  : 2019/2/18
# @Desc  : 泰坦尼克号获救问题
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold,  cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import re

titanic = pd.read_csv("titanic_train.csv")
print(titanic.head())
print(titanic.describe())
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
print(titanic.describe())
# 检查Sex特征的取值，并替换成数值
print(titanic["Sex"].unique())
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
# 检查Embarked特征的取值，并替换成数值
titanic["Embarked"] = titanic["Embarked"].fillna("S")  # 用最多的那个值进行填充
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# Initialize out algorithm class
# 线性回归
alg = LinearRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(n_splits=3, random_state=1)
predictions = []
for train_index, test_index in kf.split(titanic[predictors], titanic["Survived"]):
    print("TRAIN:", train_index, "TEST:", test_index)
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = (titanic[predictors].iloc[train_index, :])
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train_index]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test_index, :])
    predictions.append(test_predictions)
# The predictions are in three separate numpy arrays.  Concatenate them into one.
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)
# Map predictions to outcomes (only possible outcomes are 1 and 0)
predictions[predictions > .5] = 1
predictions[predictions <= .5] = 0
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)
# 逻辑回归
# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
# 随机森林
# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place
# where a tree branch ends (the bottom points of the tree)
alg = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=4, min_samples_leaf=2)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
kf = KFold(n_splits=3, random_state=1)
scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
# 创造一个家庭人员总数的列 Generating a familysize column
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
# 创建一个名字长度的列 The .apply method generates a new series
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))


def get_title(name):
    """
    A function to get the title from a name
    :param name:
    :return:
    """
    # Use a regular expression to search for a title.
    # Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


# Get all the titles and print how often each one occurs
titles = titanic["Name"].apply(get_title)
print(pd.value_counts(titles))
# Map each title to an integer. Some titles are very rare, and are compressed into the same codes as other titles
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8,
                 "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k, v in title_mapping.items():
    titles[titles == k] = v
# verify that we converted everything
print(pd.value_counts(titles))
# add in the title column
titanic["Titles"] = titles

# perform feature selection
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Titles", "NameLength"]
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# plot the scores. See which is the best
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation="vertical")
plt.show()

# Use the best 5 features
predictors = ["Pclass", "Sex", "Titles", "Fare", "NameLength"]
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
scores = cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
print(scores.mean())

# 使用多个分类器进行集合
# The algorithms we want to ensemble.
# We're using the more linear predictors for the logistic regression,
# and everything with the gradient boosting classifier.
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Titles"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Titles", "Age", "Embarked"]]
]

# Initialize the cross validation folds
kf = KFold(n_splits=3, random_state=1)

predictions = []
for train_index, test_index in kf.split(titanic[predictors], titanic["Survived"]):
    train_target = titanic["Survived"].iloc[train_index]
    full_test_predictions = []
    # Make predictions for each algorithm on each fields
    for alg, predictors in algorithms:
        # Fit the algorithm on the training data
        alg.fit(titanic[predictors].loc[train_index, :], train_target)
        # Select and predict on the test fold
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test_index, :].astype(float))[:, 1]
        full_test_predictions.append(test_predictions)
    # Use a simple ensembling scheme -- just average the predictions to get the final classification.
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # Any value over .5 is assumed to be a 1 prediction, and below .5 is a 0 prediction.
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# Put all the predictions together into one array.
predictions = np.concatenate(predictions, axis=0)

# Compute accuracy by comparing to the training data.
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)
