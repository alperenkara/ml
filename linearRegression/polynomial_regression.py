import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv("polynomial-regression.csv", sep=";")
# shape for sklearn

x = df.araba_max_hiz.values.reshape(-1,1)
y = df.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.xlabel("araba hizi")
plt.ylabel("araba hizi")
plt.show()



