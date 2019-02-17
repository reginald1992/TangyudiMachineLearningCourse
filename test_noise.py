#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_noise.py
# @Author: Shulin Liu
# @Date  : 2019/2/14
# @Desc  : 高斯噪声；瑞利噪声；伽马噪声；指数分布噪声；均匀噪声；椒盐噪声
import numpy as np
import pandas as pd
import skimage
from matplotlib import pyplot as plt
import math
import random
from scipy import stats


def wgn(x, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


N = 500
fs = 5
n = [2*math.pi*fs*t/N for t in range(N)]    # 生成了500个介于0.0-31.35之间的点
# print n
axis_x = np.linspace(0, 3, num=N)
data = [math.sin(i) for i in n]
# 添加高斯噪声
noice_gaussian1 = [random.gauss(0, 0.1) for i in range(N)]
noice_gaussian2 = [random.gauss(1, 0.5) for i in range(N)]
noice_gaussian3 = [random.gauss(2, 0.2) for i in range(N)]

voice = np.array(data)
noice_gaussian = np.array(noice_gaussian1) + np.array(noice_gaussian2) + np.array(noice_gaussian3)
voice_withnoise = voice + noice_gaussian
plt.subplot(311)
plt.plot(data)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("高斯噪声")
plt.subplot(312)
plt.plot(voice_withnoise)
plt.subplot(313)
plt.hist(noice_gaussian, bins=100)
plt.show()
# 泊松分布噪声
noice_poisson = stats.poisson.pmf(range(N), 10)
noice_poisson = np.array(noice_poisson)
voice_withnoise = voice + noice_poisson
plt.subplot(311)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("泊松噪声")
plt.plot(data)
plt.subplot(312)
plt.plot(voice_withnoise)
plt.subplot(313)
plt.hist(noice_poisson, bins=100)
plt.show()
# 瑞利噪声
a = -0.2
b = 0.03
c = [random.random() for i in range(N)]
noice_rayleigh = [a + math.sqrt(-b * math.log(1 - t)) for t in c]
noice_rayleigh = np.array(noice_rayleigh)
voice_withnoise = voice + noice_rayleigh
plt.subplot(311)
plt.plot(data)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("瑞利噪声")
plt.subplot(312)
plt.plot(voice_withnoise)
plt.subplot(313)
plt.hist(noice_rayleigh, bins=100)
plt.show()
# 伽马噪声
a = 25
b = 3
c = [random.random() for i in range(N)]
noice_erlang = [(-1/a)*math.log(1 - t) for t in c]
noice_erlang = np.array(noice_erlang)
voice_withnoise = voice + noice_erlang
plt.subplot(311)
plt.plot(data)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("伽马噪声")
plt.subplot(312)
plt.plot(voice_withnoise)
plt.subplot(313)
plt.hist(noice_erlang, bins=100)
plt.show()
# 均匀噪声
a = 0
b = 0.3
noice_uniform = [a + (b - a) * random.random() for i in range(N)]
noice_uniform = np.array(noice_uniform)
voice_withnoise = voice + noice_uniform
plt.subplot(311)
plt.plot(data)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("均匀噪声")
plt.subplot(312)
plt.plot(voice_withnoise)
plt.subplot(313)
plt.hist(noice_uniform, bins=100)
plt.show()
# 椒盐噪声
"""
椒盐噪声也成为双脉冲噪声。
在早期的印刷电影胶片上，由于胶片化学性质的不稳定和播放时候的损伤，
会使得胶片表面的感光材料和胶片的基底欠落，在播放时候，产生一些或白或黑的损伤。
事实上，这也可以归结为特殊的椒盐噪声。
椒盐噪声的实现，需要一些逻辑判断。
这里我们的思路是，产生均匀噪声，然后将超过阈值的点设置为某两个特定的幅值。
"""
a = 0.5
b = 0.5
x = [random.random() for i in range(N)]
x = np.array(x)
noice_salt_pepper = np.zeros(N)
noice_salt_pepper[x < 0.2] = -0.8
noice_salt_pepper[x > 0.8] = 0.8

voice_withnoise = voice + noice_salt_pepper
plt.subplot(311)
plt.plot(data)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.title("椒盐噪声")
plt.subplot(312)
plt.plot(voice_withnoise)
plt.subplot(313)
plt.hist(noice_salt_pepper, bins=100)
plt.show()