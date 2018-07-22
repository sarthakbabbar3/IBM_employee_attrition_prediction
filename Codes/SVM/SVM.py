# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 02:52:52 2018

@author: SARTHAK BABBAR
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import accuracy_score



df= pd.read_csv("smote.csv")

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

clf=svm.SVC()
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))