# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:26:05 2020

@author: 82103
"""

import pandas as pd
#import numpy as np
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt
import seaborn as sns

from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

df = pd.DataFrame(columns=['height', 'weight'])
df.loc[0] = [185,60]
df.loc[1] = [180,60]
df.loc[2] = [185,70]
df.loc[3] = [165,63]
df.loc[4] = [155,68]
df.loc[5] = [170,75]
df.loc[6] = [175,80]

df.head(7)


print("============================")
print(" 데이터 시각화 ")
print("============================")

sns.lmplot('height', 'weight', 
           data=df, fit_reg=False, 
           scatter_kws={"s": 200})




print("============================")
print(" k 평균 군집화 ")
print("============================")
data_points = df.values
kmeans = KMeans(n_clusters=3).fit(data_points)

kmeans.cluster_centers_

df['cluster_id'] = kmeans.labels_

df.head(12)

sns.lmplot('height', 'weight', data=df, fit_reg=False,
           scatter_kws={"s": 150},
           hue="cluster_id")


