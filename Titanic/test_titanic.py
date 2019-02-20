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
from sklearn.ensemble import RandomForestClassifier
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
