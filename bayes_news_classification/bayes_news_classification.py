#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : bayes_news_classification.py
# @Author: Shulin Liu
# @Date  : 2019/2/25
# @Desc  : 使用贝叶斯算法实现新闻分类
import pandas as pd
import jieba
"""
数据源：http://www.sogou.com/labs/resource/ca.php
"""
df_news = pd.read_table("val.txt", names=["category", "theme", "URL", "content"], en)
