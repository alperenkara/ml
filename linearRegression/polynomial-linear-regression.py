# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\PLALKAR1\Documents\GitHub\ml\linearRegression\polynomial-regression.csv',sep = ";")

y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
#plt.show()

# linear regression =  y = b0 + b1*x
# multiple linear regression   y = b0 + b1*x1 + b2*x2

# %% linear regression
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)

#%% predict
y_head = lr.predict(x)

plt.plot(x,y_head,color="red",label ="linear")
#plt.show()
speed = 10000
predictPrint = '{} lik hiz yapabilecek arabanin fiyati tahmini {}'
print(predictPrint.format(speed,lr.predict(speed)))


# %%
# polynomial regression =  y = b0 + b1*x +b2*x^2 + b3*x^3 + ... + bn*x^n
# degree = n
from sklearn.preprocessing import PolynomialFeatures
polynomial_regression = PolynomialFeatures(degree = 4)
# fit_transform -> x i mi ikinci dereceden Polynomial a cevir 
x_polynomial = polynomial_regression.fit_transform(x)


# %% fit
linear_regression2 = LinearRegression()
# y ekseni : max hiz 
# x ekseni, araba fiyatimin karesi 
linear_regression2.fit(x_polynomial,y)

# %%

y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x,y_head2,color= "green",label = "poly")
plt.legend()
plt.show()





















