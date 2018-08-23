# import data

import pandas as pd
import matplotlib.pyplot as plt 
filePath= "linear-regression-dataset.csv"
# with sep, we are seperating columns by " ; "

df = pd.read_csv(filePath,sep=";")
# plot data

plt.scatter(df.deneyim, df.maas)
plt.xlabel('deneyim')
plt.ylabel('maas')
#plt.show()

#linear regression

from sklearn.linear_model import LinearRegression
# linear regression model 
linear_reg = LinearRegression()
#transfor to numpy 
x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)
# linear regression fit
linear_reg.fit(x,y)
#prediction
b0=linear_reg.predict(0)

print(b0)