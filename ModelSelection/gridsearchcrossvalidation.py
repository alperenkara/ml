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

# grid search cross validation

from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors":np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn,grid,cv=10) # gridsearchCV
knn_cv.fit(x,y)

# print hyperparameters of KNN 
print("tuned hyperparameter K",knn_cv.best_params_)
print("the best accuracy according to tuned parameter",knn_cv.best_params_)

# grid search CV with logistic regression 

x = x[:100,:]
y = y[:100]

from sklearn.linear_model import LogisticRegression 

grid = {"C":np.logspace(-3,3,7),"penalty":["11","12"]} # L1=lasso and l2=ridge

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(x,y)

print("tuned hyperparameters: (best parameters):",logreg_cv.best_params_)

print("accuracy:",logreg_cv.best_score_)




