"""
CART: classication and regression trees
Classification
Putting splits between data clusters 
minimizing the entropy
split1 and split2
"""
import pandas as pd
import numpy as np 

data = pd.read_csv("data.csv")

data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
# 1 is good 0 zero is bad
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)
# normalization
# each column-each column's minumum
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))