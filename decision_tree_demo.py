#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/7/31 0031 20:35
# @Author : Shulin Liu
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn import tree
import pydotplus
from IPython.display import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

house = fetch_california_housing()
print(house.DESCR)
dtr = tree.DecisionTreeRegressor(max_depth=2)  # 决策树的最大深度为2
dtr.fit(house.data[:, [6, 7]], house.target)

'''
决策树可视化
'''
dot_data = tree.export_graphviz(dtr, out_file=None, feature_names=house.feature_names[6:8],
                                filled=True, impurity=False, rounded=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.get_nodes()[7].set_fillcolor("#FFF2DD")
Image(graph.create_png())
graph.write_png('dtr_white_background.png')
'''
使用决策树进行回归
'''
data_train, data_test, target_train, target_test = \
    train_test_split(house.data, house.target, test_size=0.1, random_state=16)
dtr = tree.DecisionTreeRegressor(random_state=16)
dtr.fit(data_train, target_train)
print('Decision Tree Score = ', dtr.score(data_test, target_test))
'''
使用随机森林进行回归
'''
rfr = RandomForestRegressor(random_state=16)
rfr.fit(data_train, target_train)
rfr.score(data_test, target_test)
print('Random Forest Score =', rfr.score(data_test, target_test))
'''
sklearn 导入GridSearchCV来寻找最优参数
'''
tree_param_grid = {'min_samples_split': list((3, 6, 9)), 'n_estimator': list((10, 50, 100))}
grid = GridSearchCV(RandomForestRegressor, param_grid=tree_param_grid, cv=5)
grid.fit(data_train, target_train)
print(grid.grid_scores_, grid.best_params_, grid.best_score_)
