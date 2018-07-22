# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:42:29 2018

@author: SARTHAK BABBAR
"""
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing



url = "ibm.csv"
df = pd.read_csv(url)

X=np.array(df.drop(['Attrition'],1))
y=np.array(df['Attrition'])
print(X.shape)
X=preprocessing.scale(X)
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)

print(clf.feature_importances_)  

np.savetxt('treebased.csv',clf.feature_importances_, delimiter=',')
 
