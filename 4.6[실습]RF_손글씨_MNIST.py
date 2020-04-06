# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:44:32 2020

@author: 82103
"""

from sklearn import datasets
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


mnist = datasets.load_digits()
features, labels = mnist.data, mnist.target

def cross_validation(classifier,features, labels):
    cv_scores = []

    for i in range(10):
        scores = cross_val_score(classifier, features, labels, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
    
    return cv_scores

dt_cv_scores = cross_validation(tree.DecisionTreeClassifier(), features, labels)

rf_cv_scores = cross_validation(RandomForestClassifier(), features, labels)

print("================dt_cv_scores==========================")
print(dt_cv_scores)
print("================rf_cv_scores==========================")
print(rf_cv_scores)

cv_list =   {'random_forest':rf_cv_scores,
            'decision_tree':dt_cv_scores}
          

df = pd.DataFrame(cv_list)

df.plot()

print(np.mean(dt_cv_scores))

print(np.mean(rf_cv_scores))
