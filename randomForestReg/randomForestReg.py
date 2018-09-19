import pandas as  pd
import matplotlib.pyplot as plt
import numpy as np
# importing data
df = pd.read_csv(r'C:\Users\PLALKAR1\Documents\GitHub\ml\randomForestReg\random-forest-regression-dataset.csv',sep=";",header= None)
# take whole rows and column in the zero
# convert to numpy and reshape
x = df.iloc[:,0].values.reshape(-1,1)
# for the y-index take first index
y = df.iloc[:,1].values.reshape(-1,1)

from sklearn.ensemble import RandomForestRegressor
# number of trees =100
# we have random sample chooice 
rf = RandomForestRegressor(n_estimators=100,random_state=42)
rf.fit(x,y)
value = 7.8
prediction = rf.predict(value)
print("The price is at {} state : {} ".format(value,str(prediction)))

# min value of the x to max by 0.01 steps
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
# results 
y_head = rf.predict(x_)

plt.scatter(x,y,color="blue")
plt.plot(x_,y_head,color="green")

plt.xlabel("")
plt.ylabel("")
plt.show()