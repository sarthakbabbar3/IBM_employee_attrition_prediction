# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:05:54 2018

@author: SARTHAK BABBAR
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression


df= pd.read_csv("ibm.csv")

dummy_fields = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']


for each in dummy_fields:
    dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
    df = pd.concat([df, dummies], axis=1)

fields_to_drop = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']

df = df.drop(fields_to_drop, axis=1)



X= np.array(df.drop(['Attrition'],1))

X=preprocessing.scale(X)


y=np.array(df['Attrition'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))