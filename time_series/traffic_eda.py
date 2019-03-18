#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : traffic_eda.py
# @Author: Shulin Liu
# @Date  : 2019/3/18
# @Desc  :维基百科词条EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

train = pd.read_csv('train_1.csv').fillna(0)
print(train.head())
# 将所有的数据从float转int
for col in train.columns[1:]:
    train[col] = pd.to_numeric(train[col], downcast='integer')


def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org', page)
    # print(res.group()[0:2])
    if res:
        return res.group()[0:2]
    return 'na'


train['lang'] = train.Page.map(get_language)
print(Counter(train.lang))
lang_sets = {'en': train[train.lang == 'en'].iloc[:, 0:-1], 'ja': train[train.lang == 'ja'].iloc[:, 0:-1],
             'de': train[train.lang == 'de'].iloc[:, 0:-1], 'na': train[train.lang == 'na'].iloc[:, 0:-1],
             'fr': train[train.lang == 'fr'].iloc[:, 0:-1], 'zh': train[train.lang == 'zh'].iloc[:, 0:-1],
             'ru': train[train.lang == 'ru'].iloc[:, 0:-1], 'es': train[train.lang == 'es'].iloc[:, 0:-1]}
sums = {}
for key in lang_sets:
    sums[key] = lang_sets[key].iloc[:, 1:].sum(axis=0) / lang_sets[key].shape[0]
days = [r for r in range(sums['en'].shape[0])]

fig = plt.figure(1, figsize=[10, 10])
plt.ylabel('Views per Page')
plt.xlabel('Day')
plt.title('Pages in Different Languages')
labels = {'en': 'English', 'ja': 'Japanese', 'de': 'German',
          'na': 'Media', 'fr': 'French', 'zh': 'Chinese',
          'ru': 'Russian', 'es': 'Spanish'
          }
for key in sums:
    plt.plot(days, sums[key], label=labels[key])
plt.legend()
plt.show()


def plot_entry(key, idx):
    data = lang_sets[key].iloc[idx, 1:]
    fig = plt.figure(1, figsize=(10, 5))
    plt.plot(days, data)
    plt.xlabel('day')
    plt.ylabel('views')
    plt.title(train.iloc[lang_sets[key].index[idx], 0])
    plt.show()


idx = [1, 5, 10, 50, 100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000]
for i in idx:
    plot_entry('en', i)

npages = 5
top_pages = {}
for key in lang_sets:
    print(key)
    sum_set = pd.DataFrame(lang_sets[key][['Page']])
    sum_set['total'] = lang_sets[key].sum(axis=1)
    sum_set = sum_set.sort_values('total', ascending=False)
    print(sum_set.head(10))
    top_pages[key] = sum_set.index[0]
    print('\n\n')
for key in top_pages:
    fig = plt.figure(1, figsize=(10, 5))
    cols = train.columns
    cols = cols[1:-1]
    data = train.loc[top_pages[key], cols]
    plt.plot(days, data)
    plt.xlabel('Days')
    plt.ylabel('Views')
    plt.title(train.loc[top_pages[key], 'Page'])
    plt.show()
