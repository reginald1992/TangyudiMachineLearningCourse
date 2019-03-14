#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_xgboost.py
# @Author: Shulin Liu
# @Date  : 2019/3/14
# @Desc  : Xgboost使用实例
import xgboost
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
# load data
dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split data into x and y
x = dataset[:, 0:8]
y = dataset[:, 8]
# split data into train and test
seed = 7
test_size = 0.33
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
# fit the model
model = xgboost.XGBClassifier()
model.fit(x_train, y_train)
# make prediction
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
# evaluate prediction
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: %.2f%%' % (accuracy * 100.0))
# ---------------------------------------------------------
eval_set = [(x_test, y_test)]
model.fit(x_train, y_train, early_stopping_rounds=10, eval_metric='logloss', eval_set=eval_set, verbose=True)
# make predictions for test data
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
# plot feature importance
xgboost.plot_importance(model)
plt.show()
# Xgboost 参数调节
'''
1.learning rate
2.tree 
max_depth
min_child_weight
subsample, colsample_bytree
gamma 
3.正则化参数
lambda 
alpha 
'''
xgb1 = xgboost.XGBClassifier(
 learning_rate=0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective='binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
# tune learning rate
model = xgboost.XGBClassifier()
learning_rate = [0.0001, 0.001, 0.01, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring='neg_log_loss', n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(x, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean, param in zip(means, params):
    print("%f  with: %r" % (mean, param))
