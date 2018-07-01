"""
数据描述：
我们将建立一个逻辑回归模型来预测一个学生是否被大学录取。假设你是一个大学系的管理员，
你想根据两次考试的结果来决定每个申请人的录取机会。你有以前的申请人的历史数据，你可以用它作为逻辑回归的训练集。
对于每一个培训例子，你有两个考试的申请人的分数和录取决定。为了做到这一点，我们将建立一个分类模型，根据考试成绩估计入学概率。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
path = 'data' + os.sep + 'LogiReg_data.txt'
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])


