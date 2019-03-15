#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ml_7_mulabel.py
# @Author: Shulin Liu
# @Date  : 2019/3/15
# @Desc  : 多类别问题
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "year", "origin", "car name"]
cars = pd.read_table("auto-mpg.data", delim_whitespace=True, names=columns)
print(cars.head(5))
print(cars.tail(5))

dummy_cylinders = pd.get_dummies(cars["cylinders"], prefix="cyl")
print(dummy_cylinders)
cars = pd.concat([cars, dummy_cylinders], axis=1)
print(cars.head())
dummy_years = pd.get_dummies(cars["year"], prefix="year")
print(dummy_years)
cars = pd.concat([cars, dummy_years], axis=1)
cars = cars.drop("year", axis=1)
cars = cars.drop("cylinders", axis=1)
print(cars.head())

shuffled_rows = np.random.permutation(cars.index)
shuffled_cars = cars.iloc[shuffled_rows]
highest_train_row = int(cars.shape[0] * .70)
train = shuffled_cars.iloc[0:highest_train_row]
test = shuffled_cars.iloc[highest_train_row:]

unique_origins = cars["origin"].unique()
unique_origins.sort()

models = {}
features = [c for c in train.columns if c.startswith("cyl") or c.startswith("year")]

for origin in unique_origins:
    model = LogisticRegression()

    X_train = train[features]
    y_train = train["origin"] == origin

    model.fit(X_train, y_train)
    models[origin] = model

testing_probs = pd.DataFrame(columns=unique_origins)
print(testing_probs)

for origin in unique_origins:
    # Select testing features.
    X_test = test[features]
    # Compute probability of observation being in the origin.
    testing_probs[origin] = models[origin].predict_proba(X_test)[:, 1]
print(testing_probs)
