import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np 	 
data = pd.read_csv(r"C:\Users\\alper\Documents\GitHub\ml\KNN\data.csv")

# malignant = M, kotu huylu in Turkish language
# benign = B, iyi huylu in Turkish language
# we are going to erase ID and
# axis=1 takes whole column
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
print(data.tail())
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
# alpha provides tranparency
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="Malignant",alpha=0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="Benign",alpha=0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
#plt.show()
# 
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)


# normalization

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
# train and test splitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3 ,random_state=1)
# KNN Model
from sklearn.neighbors import KNeighborsClassifier
# k = 3
n_neighbors=3
knn = KNeighborsClassifier(n_neighbors)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
score = knn.score(x_test,y_test)
print("KNN Score {} with K: {} ".format(score,n_neighbors))

# find the best k value

