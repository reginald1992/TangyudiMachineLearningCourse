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
import gensim
from sklearn.model_selection import train_test_split

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
dictionary = gensim.corpora.Dictionary(content_clean)
# 做映射，相当于词袋
corpus = [dictionary.doc2bow(sentence) for sentence in content_clean]
# 类似于Kmeans自己指定K值
lda = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
print(lda.print_topic(1, topn=5))
for topic in lda.print_topics(num_topics=20, num_words=5):
    print(topic[1])
# 贝叶斯分类器做新闻分类
df_train = pd.DataFrame({"contents_clean": content_clean, "label": df_news["category"]})
print(df_train.tail())
print(df_train.label.unique())
label_mapping = {'汽车': 0, '财经': 1, '科技': 2, '健康': 3, '体育': 4, '教育': 5, '文化': 6, '军事': 7, '娱乐': 8, '时尚': 9}
df_train["label"] = df_train["label"].map(label_mapping)
x_train, x_test, y_train, y_test = train_test_split(df_train["contents_clean"].values, df_train["label"].values,
                                                    random_state=1)
# 向量化
words = []
for line_index in range(len(x_train)):
    try:
        words.append(" ".join(x_train[line_index]))
    except:
        print(line_index)
print(words[0])

# 向量化举例
from sklearn.feature_extraction.text import CountVectorizer
texts = ["dog cat fish", "dog cat cat", "fish bird", 'bird']
cv = CountVectorizer()
cv_fit = cv.fit_transform(texts)
print(cv.get_feature_names())
print(cv_fit.toarray())
print(cv_fit.toarray().sum(axis=0))
# 对新闻数据进行向量化
vec = CountVectorizer(analyzer='word', max_features=4000, lowercase=False)
vec.fit(words)
# 使用贝叶斯进行分类
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(vec.transform(words), y_train)
# 对测试集进行同样的处理
test_words = []
for line_index in range(len(x_test)):
    try:
        test_words.append(" ".join(x_test[line_index]))
    except:
        print(line_index)
# 进行测试并打分
result = classifier.score(vec.transform(test_words), y_test)
print(result)
