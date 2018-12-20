# choose the k value
# two random centeroid 
# assign data points to centeroids according to distance
# WCSS method
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# creating dataset

# k=3, three different cluster
# gaussian variables
# avearage 25 , 5 sigma ,1000 variables

# class1
x1 = np.random.normal(25,5,1000)
y1 = np.random.normal(25,5,1000)

# class2
x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(60,5,1000)

# class3

x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)

x = np.concatenate((x1,x2,x3),axis=0)
y = np.concatenate((y1,y2,y3),axis=0)

dictionary = {"x":x,"y":y}

data = pd.DataFrame(dictionary)

#plt.scatter(x1,y1,color="black")
#plt.scatter(x2,y2,color="black")
#plt.scatter(x3,y3,color="black")
#plt.show()

from sklearn.cluster import KMeans
wcss = []

for k in range(1,15):
    # to finding best k value
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    # inertia->wcss
    wcss.append(kmeans.inertia_)


#plt.plot(range(1,15),wcss)
#plt.xlabel("cluster value")
#plt.ylabel("wcss")
#plt.show()


kmeans2 = KMeans(n_clusters=3)
clusters = kmeans2.fit_predict(data)

data["label"] = clusters

plt.scatter(data.x[data.label == 0],data.y[data.label == 0],color = "red")
plt.scatter(data.x[data.label == 1],data.y[data.label == 1],color = "green")
plt.scatter(data.x[data.label == 2],data.y[data.label == 2],color = "blue")
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1],color = "yellow")

plt.show()