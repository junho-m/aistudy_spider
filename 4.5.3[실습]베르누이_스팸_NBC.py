# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:50:19 2020

@author: 82103
"""


import numpy as np
import pandas as pd

# 베르누이 나이브베이즈를 위한 라이브러리를 임포트합니다
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

# 모델의 정확도 평가를 위해 임포트합니다
from sklearn.metrics import accuracy_score


print("=========================================================")
print(" 데이터 수집 ");
print("=========================================================")

email_list = [
                {'email title': 'free game only today', 'spam': True},
                {'email title': 'cheapest flight deal', 'spam': True},
                {'email title': 'limited time offer only today only today', 'spam': True},
                {'email title': 'today meeting schedule', 'spam': False},
                {'email title': 'your flight schedule attached', 'spam': False},
                {'email title': 'your credit card statement', 'spam': False}
             ]
df = pd.DataFrame(email_list)

print(df)

print("=========================================================")
print(" 데이터 다듬기 ");
print("=========================================================")

df['label'] = df['spam'].map({True:1,False:0})

print(df)

# 학습에 사용될 데이터와 분류값을 나눕니다
df_x=df["email title"]
df_y=df["label"]

cv = CountVectorizer(binary=True)
x_traincv=cv.fit_transform(df_x)

encoded_input=x_traincv.toarray()
encoded_input

cv.inverse_transform(encoded_input[0])

cv.get_feature_names()

print("=========================================================")
print(" 베르누이 나이브베이즈 분류 ");
print("=========================================================")

# 학습 데이터로 베르누이 분류기를 학습합니다
bnb = BernoulliNB()
y_train=df_y.astype('int')
bnb.fit(x_traincv,y_train)
print(bnb)

print("=========================================================")
print(" 테스트 데이터 다듬기 ");
print("=========================================================")

test_email_list = [
                {'email title': 'free flight offer', 'spam': True},
                {'email title': 'hey traveler free flight deal', 'spam': True},
                {'email title': 'limited free game offer', 'spam': True},
                {'email title': 'today flight schedule', 'spam': False},
                {'email title': 'your credit card attached', 'spam': False},
                {'email title': 'free credit card offer only today', 'spam': False}
             ]
test_df = pd.DataFrame(test_email_list)
test_df['label'] = test_df['spam'].map({True:1,False:0})
test_x=test_df["email title"]
test_y=test_df["label"]
x_testcv=cv.transform(test_x)
print(x_testcv)

print("=========================================================")
print(" 테스트 ");
print("=========================================================")

predictions=bnb.predict(x_testcv)
print(predictions)

print("=========================================================")
print(" 정확도 ");
print("=========================================================")

print(accuracy_score(test_y, predictions))









