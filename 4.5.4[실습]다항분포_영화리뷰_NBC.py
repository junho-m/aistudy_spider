# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 14:23:00 2020

@author: 82103
"""

#import numpy as np
import pandas as pd

# 다항분포 나이브베이즈를 위한 라이브러리를 임포트합니다
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 모델의 정확도 평가를 위해 임포트합니다
from sklearn.metrics import accuracy_score



review_list = [
                {'movie_review': 'this is great great movie. I will watch again', 'type': 'positive'},
                {'movie_review': 'I like this movie', 'type': 'positive'},
                {'movie_review': 'amazing movie in this year', 'type': 'positive'},
                {'movie_review': 'cool my boyfriend also said the movie is cool', 'type': 'positive'},
                {'movie_review': 'awesome of the awesome movie ever', 'type': 'positive'},
                {'movie_review': 'shame I wasted money and time', 'type': 'negative'},
                {'movie_review': 'regret on this move. I will never never what movie from this director', 'type': 'negative'},
                {'movie_review': 'I do not like this movie', 'type': 'negative'},
                {'movie_review': 'I do not like actors in this movie', 'type': 'negative'},
                {'movie_review': 'boring boring sleeping movie', 'type': 'negative'}
             ]
df = pd.DataFrame(review_list)
print(df)

df['label'] = df['type'].map({"positive":1,"negative":0})
print(df)

df_x=df["movie_review"]
df_y=df["label"]

cv = CountVectorizer()

x_traincv=cv.fit_transform(df_x)
encoded_input=x_traincv.toarray()

print(encoded_input)

cv.inverse_transform(encoded_input[0])

cv.get_feature_names()



# 기존의 데이터로 학습을 진행합니다
mnb = MultinomialNB()
y_train=df_y.astype('int')
mnb.fit(x_traincv,y_train)

test_feedback_list = [
                {'movie_review': 'great great great movie ever', 'type': 'positive'},
                {'movie_review': 'I like this amazing movie', 'type': 'positive'},
                {'movie_review': 'my boyfriend said great movie ever', 'type': 'positive'},
                {'movie_review': 'cool cool cool', 'type': 'positive'},
                {'movie_review': 'awesome boyfriend said cool movie ever', 'type': 'positive'},
                {'movie_review': 'shame shame shame', 'type': 'negative'},
                {'movie_review': 'awesome director shame movie boring movie', 'type': 'negative'},
                {'movie_review': 'do not like this movie', 'type': 'negative'},
                {'movie_review': 'I do not like this boring movie', 'type': 'negative'},
                {'movie_review': 'aweful terrible boring movie', 'type': 'negative'}
             ]
test_df = pd.DataFrame(test_feedback_list)
test_df['label'] = test_df['type'].map({"positive":1,"negative":0})
test_x=test_df["movie_review"]
test_y=test_df["label"]

# 테스트를 진행합니다
x_testcv=cv.transform(test_x)
predictions=mnb.predict(x_testcv)

print(accuracy_score(test_y, predictions))






