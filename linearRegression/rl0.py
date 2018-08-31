# import data
from sklearn.linear_model import LinearRegression
import numpy as np
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


# linear regression model 
linear_reg = LinearRegression()
#transfor to numpy and make all types same !
x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)
# linear regression fit
linear_reg.fit(x,y)
#prediction
b0=linear_reg.predict(0)
print("b0: ", b0)
b0_ = linear_reg.intercept_
print("b0 intercept: ", b0_)
# print value of slope
b1 = linear_reg.coef_
print("b1: ", b1)

#salary prediction
deneyim_deger = 11 # how many years experience you have
maas_yeni = np.round((b0 + b1*deneyim_deger),2)
print("Salary expection with {} years experience is {}".format(deneyim_deger,maas_yeni))

 sdf3
print(linear_reg.predict(11))

array = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]).reshape(-1,1)#deneyim

plt.scatter(x,y)

y_head = linear_reg.predict(array)

plt.plot(array, y_head, color="blue")
plt.show()
