# emsamble learning model
# gathering the decision trees 
# choose N sample from train data(subSample)
# train with decision tree model
# again split the train to subSample

import pandas as pd
import numpy as np
import os 

data = pd.read_csv("D:\mygit\ml\\randomForestClass\data.csv")

data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

y = data.diagnosis.values

x_data = data.drop(["diagnosis"],axis=1)

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

# train and test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.15, random_state =42)

from sklearn.ensemble import RandomForestClassifier 

rf = RandomForestClassifier(n_estimators=10, random_state= 1)

rf.fit(x_train,y_train)

print("random forest algorithm result {} ".format(rf.score(x_test,y_test)))



