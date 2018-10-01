# performance model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(r"C:\Users\PLALKAR1\Documents\GitHub\ml\evaluationRegModels\randomForestReg.csv",sep=";",header=None)
# axis
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
# 100 trees, same randomness with 42, creating sub-data
rf = RandomForestRegressor(n_estimators=100, random_state= 42)
rf.fit(x,y)
# using the same data which we already trained
# we need a test data as separately
# y_head is our line which we fit on the graph 
y_head = rf.predict(x)

from sklearn.metrics import r2_score
# y-> real y values 
# y_head predicted values
print('r_score: {}'.format(r2_score(y,y_head)))