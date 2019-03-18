#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : robot_failure_example.py
# @Author: Shulin Liu
# @Date  : 2019/3/18
# @Desc  : 对时间序列做分类，使用tsfresh库
import matplotlib.pylab as plt
import seaborn as sns
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
if __name__ == '__main__':
    download_robot_execution_failures()
    df, y = load_robot_execution_failures()
    print(df.head())
    df[df.id == 3][['time', 'F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']].plot(x='time', title='Success example (id 3)', figsize=(12, 6))
    plt.show()
    df[df.id == 20][['time', 'F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']].plot(x='time', title='Failure example (id 20)', figsize=(12, 6))
    plt.show()
    extraction_settings = ComprehensiveFCParameters()
    # column_id (str) – The name of the id column to group by
    # column_sort (str) – The name of the sort column.
    X = extract_features(df, column_id='id', column_sort='time', default_fc_parameters=extraction_settings, impute_function=impute)
    print(X.head())
    print(X.info())
    X_filtered = extract_relevant_features(df, y, column_id='id', column_sort='time', default_fc_parameters=extraction_settings)
    print(X_filtered.head())
    print(X_filtered.info())
    X_train, X_test, X_filtered_train, X_filtered_test, y_train, y_test = train_test_split(X, X_filtered, y,test_size=0.4)
    cl = DecisionTreeClassifier()
    cl.fit(X_train, y_train)
    print(classification_report(y_test, cl.predict(X_test)))
    print(cl.n_features_)
    cl2 = DecisionTreeClassifier()
    cl2.fit(X_filtered_train, y_train)
    print(classification_report(y_test, cl2.predict(X_filtered_test)))
    print(cl2.n_features_)




