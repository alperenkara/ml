import pandas as pd

df = pd.read_csv("linear-regression-dataset.csv")

df2 = pd.DataFrame(df,columns = ['deneyim','maas'])

print(df2)