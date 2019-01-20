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
import time

path = 'data' + os.sep + 'LogiReg_data.txt'
pdData = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
print(pdData.head())
print(pdData.shape)
positive = pdData[pdData['Admitted'] == 1]
negative = pdData[pdData['Admitted'] == 0]
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=100, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=100, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()
'''
逻辑回归
目标：建立分类器（求解三个参数θ0、θ1、θ2）
设定阈值：根据阈值判断录取结果
由以下几个模块组成：
（1）sigmoid:映射到概率的函数
（2）model:返回预测结果值
（3）cost：根据参数计算损失
（4）gradient：计算每个参数的梯度方向
（5）descent：进行参数更新
（6）accuracy：计算精度
'''


def sigmoid(z):
    """
    :param z: 原始数据
    :return: SIGMOD函数下的概率值
    """
    return 1 / (1 + np.exp(-z))


nums = np.arange(-10, 10, step=1)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()


def model(x, theta):
    """
    :param x:
    :param theta:
    :return:
    """
    return sigmoid(np.dot(x, theta.T))


pdData.insert(0, 'Ones', 1)
# in a try / except structure so as not to return an error if the block si executed several times
# set X (training data) and y (target variable)
originalData = pdData.as_matrix()  # convert the Pandas representation of the data to an array useful for computations
cols = originalData.shape[1]
x = originalData[:, 0:cols - 1]
y = originalData[:, cols - 1: cols]
theta = np.zeros([1, 3])


def cost(x, y, theta):
    """
    损失函数：将对数似然函数去负号，并求平均损失
    :param x:
    :param y:
    :param theta:
    :return:
    """
    left = np.multiply(-y, np.log(model(x, theta)))
    right = np.multiply(1 - y, np.log(1 - model(x, theta)))
    return np.sum(left - right) / (len(x))


def gradient(x, y, theta):
    """
    计算梯度
    :param x:
    :param y:
    :param theta:
    :return:
    """
    grad = np.zeros(theta.shape)
    error = (model(x, theta) - y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, x[:, j])
        grad[0, j] = np.sum(term) / len(x)
    return grad


'''
Gradient descent
比较三种不同梯度下降的方法
'''
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2


def stop_criterion(type, value, threshold):
    if type == STOP_ITER:
        return value > threshold
    elif type == STOP_COST:
        return abs(value[-1] - value[-2]) < threshold
    elif type == STOP_GRAD:
        return np.linalg.norm(value) < threshold


def shuffle_data(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    x = data[:, 0: cols - 1]
    y = data[:, cols - 1:]
    return x, y


def descent(data, theta, batch_size, stop_type, thresh, alpha):
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    x, y = shuffle_data(data)
    grad = np.zeros(theta.shape)
    costs = [cost(x, y, theta)]

    while True:
        grad = gradient(x[k: k + batch_size], y[k: k + batch_size], theta)
        k += batch_size  # 取batch数量个数据
        if k >= n:
            k = 0
            x, y = shuffle_data(data)  # 重新洗牌
            theta = theta - alpha * grad  # 参数更新
            costs.append(cost(x, y, theta))  # 计算新的损失
            i += 1

            if stop_type == STOP_ITER:
                value = i
            elif stop_type == STOP_COST:
                value = costs
            elif stop_type == STOP_GRAD:
                value = grad
            if stop_criterion(stop_type, value, thresh):
                break
    return theta, i - 1, costs, grad, time.time() - init_time


def run_expe(data, theta, batch_size, stop_type, thresh, alpha):
    theta, iter, costs, grad, dur = descent(data, theta, batch_size, stop_type, thresh, alpha)
    name = "Original" if (data[:, 1] > 2).sum() > 1 else "Scaled"
    name += "data - learning rate: {} - ".format(alpha)
    if batch_size == n:
        str_desc_type = "Gradient"
    elif batch_size == 1:
        str_desc_type = "Stochastic"
    else:
        str_desc_type = "Mini-batch({})".format(batch_size)
    name += str_desc_type + " descent - Stop:"
    if stop_type == STOP_ITER:
        str_stop = "{} iterations".format(thresh)
    elif stop_type == STOP_COST:
        str_stop = "cost change < {}".format(thresh)
    elif stop_type == STOP_GRAD:
        str_stop = "gradient norm < {}".format(thresh)
    name += str_stop
    print("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title(name.upper() + '- Error vs. Iteration')
    plt.show()
    return theta


'''
不同的停止策略
'''
# 设定迭代次数
n = 100
run_expe(originalData, theta, n, STOP_ITER, thresh=5000, alpha=0.000001)

# 根据损失值停止
run_expe(originalData, theta, n, STOP_COST, thresh=0.000001, alpha=0.001)

# 根据梯度变化停止
run_expe(originalData, theta, n, STOP_GRAD, thresh=0.05, alpha=0.001)

'''
对比不同的梯度下降方法
'''
# Stochastic descent
run_expe(originalData, theta, 1, STOP_ITER, thresh=5000, alpha=0.001)
# 把学习率调小
run_expe(originalData, theta, 1, STOP_ITER, thresh=15000, alpha=0.000002)

# Mini-batch descent
run_expe(originalData, theta, 16, STOP_ITER, thresh=15000, alpha=0.001)

# 浮动仍然比较大，我们来尝试下对数据进行标准化 将数据按其属性(按列进行)减去其均值，然后除以其方差。最后得到的结果是，
# 对每个属性/每列来说所有数据都聚集在0附近，方差值为1
from sklearn import preprocessing as pp

scaled_data = originalData.copy()
scaled_data[:, 1:3] = pp.scale(originalData[:, 1:3])
run_expe(scaled_data, theta, n, STOP_ITER, thresh=15000, alpha=0.001)

run_expe(scaled_data, theta, n, STOP_GRAD, thresh=0.02, alpha=0.001)
# 更多的迭代次数
run_expe(scaled_data, theta, n, STOP_GRAD, thresh=0.02/5, alpha=0.001)
# 随机梯度下降更快，但是我们需要迭代的次数也需要更多，所以还是用batch的比较合适！！！
run_expe(scaled_data, theta, 16, STOP_GRAD, thresh=0.02*2, alpha=0.001)


def predict(x, theta):
    """
    设定阈值和精度
    :param x:
    :param theta:
    :return:
    """
    return [1 if y >= 0.5 else 0 for y in model(x, theta)]


scaled_X = scaled_data[:, :3]
y = scaled_data[:, 3]
predictions = predict(scaled_X, theta)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))
