# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 00:11:30 2018

@author: SARTHAK BABBAR
"""

import numpy as np
import warnings
from collections import Counter
import pandas as pd
import random 

def k_nearest_neighbors(data,predict,k=3):
    if len(data) >=k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances=[]
    for group in data:
        for features in data[group]:
            euclidean_distance= np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
             
    votes=[i[1] for i in sorted(distances)[:k]]
    #contains sorted classes since we are calculating i[1] which is class and i[0] is distance
    #print(Counter(votes).most_common(1))
    votes_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    #print(votes_result,confidence)
    
    return votes_result,confidence

accuracies = []
for i in range(5):
    #df=pd.read_csv('ibm.csv')
    df=pd.read_csv('smote.csv')
    df.replace('?',-99999,inplace=True)
    #df.drop(['id'],1,inplace=True)
    full_data=df.astype(float).values.tolist()
    
    random.shuffle(full_data)
    
    test_size=0.4
    train_set = {1:[] , 0:[]}
    test_set={1:[] , 0:[]}
    train_data=full_data[:-int(test_size*len(full_data))]
    test_data=full_data[-int(test_size*len(full_data)):]
    
    #i[-1] is the last column
    
    for i in train_data: 
        train_set[i[-1]].append(i[:-1])
    
    for i in test_data:    
        test_set[i[-1]].append(i[:-1])
    
    
    correct=0
    total=0
    
    for group in test_set:
        for data in test_set[group]:
            vote,confidence = k_nearest_neighbors(train_set,data,k=5)
            if group == vote:
                correct+=1
            #else: 
                #print(confidence)
            total+=1
            
    #print('Accuracy :', correct/total)
    accuracies.append(correct/total)
    
print(sum(accuracies)/len(accuracies))



