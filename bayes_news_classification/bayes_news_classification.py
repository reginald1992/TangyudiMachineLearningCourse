#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : bayes_news_classification.py
# @Author: Shulin Liu
# @Date  : 2019/2/25
# @Desc  : 使用贝叶斯算法实现新闻分类
import pandas as pd
import jieba
import numpy as np
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import matplotlib
import jieba.analyse

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
    """
    去停用词函数
    :param contents:
    :param stopwords:
    :return:去除停用词的内容和所有的非停用词
    """
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


contents = df_content.content_s.values.tolist()
stopwords = stopwords.stopwords.values.tolist()
content_clean, all_worsd = drop_stopwords(content, stopwords)
df_content = pd.DataFrame({"content_clean": content_clean})
df_all_words = pd.DataFrame({"all_words": all_worsd})
# 统计词频
words_count = df_all_words.groupby(by=["all_words"])["all_words"].agg({"count": np.size})
words_count = words_count.reset_index().sort_values(by=["count"], ascending=False)
# 画出词云
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
wordcloud = WordCloud(font_path="simhei.ttf", background_color="white", max_font_size=80)
word_frequence = {x[0]: x[1] for x in words_count.head(100).values}
wordcloud = wordcloud.fit_words(word_frequence)
plt.imshow(wordcloud)
plt.show()
# TF—IDF 提取关键词
index = 2000
print(df_news["content"][index])
content_s_str = "".join(content_s[index])
print(" ".join(jieba.analyse.extract_tags(content_s_str, topK=5, withWeight=False)))
# LDA: 主题模型
# 格式要求：list of list形式，分词好的整个语料
