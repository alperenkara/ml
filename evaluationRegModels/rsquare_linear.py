import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\PLALKAR1\Documents\GitHub\ml\evaluationRegModels\linearRegData.csv",sep=";")


plt.scatter(df.deneyim,df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()


from sklearn.linear_model import LinearRegression
# linear regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1)
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x,y)

y_head = linear_reg.predict(x) # prediction of salary

plt.plot(x, y_head, color = "red")