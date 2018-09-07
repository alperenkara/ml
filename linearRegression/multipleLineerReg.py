# firstly we should prepare our dataset

# salary is effected by age and experience

# our main aim is that reaching minumum mean square error
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
# ; is my seperator
<<<<<<< HEAD
df = pd.read_csv("multiple-linear-regression-dataset.csv", sep=";")
=======
df = pd.read_csv(r'C:\Users\PLALKAR1\Documents\GitHub\ml\linearRegression\multiple-linear-regression-dataset.csv',sep=";")
>>>>>>> ae7525116984bbefcbbfaeb303c4e9df87b627e2
print(df)
# whole rows and take only zero and second column
x = df.iloc[:, [0, 2]].values
y = df.maas.values.reshape(-1, 1)

multiple_linear_regression = LinearRegression()
# give us a line fit with x and y values
multiple_linear_regression.fit(x, y)

# b1 constant/bias
# maas_yeni = np.round((b0 + b1*deneyim_deger),2)
print("b0: ", multiple_linear_regression.intercept_)
print("b1, b2: ", multiple_linear_regression.coef_)

# make prediction
# we have experience and age for future values
# first value is experience and second values is age
a = multiple_linear_regression.predict(np.array([[10, 35], [5, 35]]))

print(a)
