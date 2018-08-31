# firstly we should prepare our dataset

# salary is effected by age and experience

# our main aim is that reaching minumum mean square error
#  import pandas as import pd

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
# ; is my seperator
df = pd.read_csv("multiple-linear-regression-dataset.csv",sep=";")
print(df)
# whole rows and take only zero and second column 
x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)