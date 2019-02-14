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
noice_gaussian = [random.gauss(0, 0.1) for i in range(N)]
voice = np.array(data)
noice_gaussian = np.array(noice_gaussian)
voice_withnoise = voice + noice_gaussian
plt.subplot(311)
plt.plot(data)
plt.subplot(312)
plt.plot(voice_withnoise)
plt.subplot(313)
plt.plot(noice_gaussian)
plt.show()
# 泊松分布噪声
noice_poisson = stats.poisson.pmf(range(N), 0.5)
noice_poisson = np.array(noice_poisson)
voice_withnoise = voice + noice_poisson
plt.subplot(311)
plt.plot(data)
plt.subplot(312)
plt.plot(voice_withnoise)
plt.subplot(313)
plt.plot(noice_gaussian)
plt.show()
# 瑞利噪声
a = -0.2
b = 0.03
noice_rayleigh = [a + math.sqrt(-b * math.log(1 - np.random.rand(t))) for t in range(N)]

