# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 19:57:55 2018

@author: SARTHAK BABBAR
"""

import pandas as pd 
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing

df= pd.read_csv("ibm.csv")
df=df.dropna(axis=0,how='any')

print(df.shape)
X=np.array(df.drop(['Attrition'],1))


y=np.array(df['Attrition'])

sm = SMOTE()

X_resampled,y_resampled = sm.fit_sample(X, y)

print(X_resampled.shape)

np.savetxt('smotex.csv',X_resampled, delimiter=',')
np.savetxt('smotey.csv',y_resampled, delimiter=',')
   


