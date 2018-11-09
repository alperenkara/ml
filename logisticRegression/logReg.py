# binary classification
# cat or dog, zero or one
# transforming to numpy array
# input, weights, bias , sigmoid function

# learning rate must have choosen wisely
# hyperparameter -> tuned parameters

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("D:\mygit\\ml\\logisticRegression\data.csv")
# erasing unnecessary columns
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
# print(data.info())
# column of diagnosis turns binary
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
# on the y=axis there is only diagnosis values
y = data.diagnosis.values
# on the x-axis there will be everything except the diagnosis
x_data = data.drop(["diagnosis"], axis=1)

# normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values

# train test split
# %80 is train set %20 test set

