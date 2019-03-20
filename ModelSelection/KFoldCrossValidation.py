# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:24:19 2019

@author: alperen
"""

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

iris = load_iris()

x = iris.data # features
y = iris.target

# normalization

x = (x-np.min(x))/(np.max(x)-np.min(x))

# split the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(x,y,test_size=0.3)

# KNN MOdel

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)

# K Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = knn, X = x_train, y = y_train, cv=10)

print("average accuracy: ",np.mean(accuracies))
print("average std: ",np.std(accuracies))

# Real Test
knn.fit(x_train,y_train)
print("test accuracy: ", knn.score(x_test,y_test))

