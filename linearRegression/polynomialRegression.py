import matplotlib.pyplot as plt 
import pandas as pd
df = pd.read_csv("polynomial_regression.csv",sep=';')
# shape for sklearn

x = df.araba_max_hiz.values.reshape(-1,1)
y = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("araba hizi")
plt.ylabel("araba hizi")
plt.show()

# linear regression : y = b0 + b1*x
# multiple linear regression : y = b0*x1 +b2*x2

# %% linear regression 
from sklearn.linear_model import LinearRegression 

lr = LinearRegression()

lr.fit(x,y)

# prediction

y_head = lr.predict(x)

plt.plot(x,y_head,color='red')
plt.show()

lr.predict(10000)



