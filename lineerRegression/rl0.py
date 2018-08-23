import pandas as pd
import matplotlib.pyplot as plt 
filePath= "linear-regression-dataset.csv"

df = pd.read_csv(filePath,sep=";")

plt.scatter(df.deneyim, df.maas)
plt.xlabel('deneyim')
plt.ylabel('maas')
plt.show()