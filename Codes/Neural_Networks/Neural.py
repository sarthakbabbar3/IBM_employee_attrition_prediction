# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:52:36 2018

@author: SARTHAK BABBAR
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation,preprocessing
from my_answers import NeuralNetwork


def MSE(y, Y):
    return np.mean((y-Y)**2)

data_path = 'ibm.csv'
df = pd.read_csv(data_path)
#print(df.head())

dummy_fields = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
for each in dummy_fields:
    dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
    df = pd.concat([df, dummies], axis=1)

fields_to_drop = ['Department', 'EducationField', 'JobRole', 'MaritalStatus']
data = df.drop(fields_to_drop, axis=1)
#print(data.shape)


X= np.array(data.drop(['Attrition'],1))
X=preprocessing.scale(X) 
y=np.array(data['Attrition'])

X_train, test_features, y_train, test_targets = cross_validation.train_test_split(X,y, test_size=0.2)

train_features,val_features, train_targets, val_targets = cross_validation.train_test_split(X_train,y_train,test_size=0.2)

print(train_features.shape)


import sys

from my_answers import iterations, learning_rate, hidden_nodes, output_nodes


N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    #batch = np.random.choice(train_features, size=128)
    X, y = train_features, train_targets
                             
    network.train(X, y)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets)
    val_loss = MSE(network.run(val_features).T, val_targets)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)
