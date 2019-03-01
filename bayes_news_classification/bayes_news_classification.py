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
df_news = pd.read_table("val.txt", names=["category", "theme", "URL", "content"], encoding='utf-8')
df_news = df_news.dropna()
print(df_news.head())
content = df_news.content.values.tolist()
print(content[1000])
# 使用结巴分词器进行分词
content_s = []
for line in content:
    current_segment = jieba.lcut(line)
    if len(current_segment) > 1 and current_segment != '\r\n':
        content_s.append(current_segment)
print(content_s[1000])
df_content = pd.DataFrame({"content_s": content_s})
# 去停用词
stopwords = pd.read_csv("stopwords.txt", index_col=False, sep='\t', quoting=3, names=["stopwords"], encoding='utf-8')


def drop_stopwords(contents, stopwords):
    content_clean = []
    all_words = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
            all_words.append(str(word))
        content_clean.append(line_clean)
    return content_clean, all_words


