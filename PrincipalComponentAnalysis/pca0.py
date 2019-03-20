# Principle Component Analysis
"""
1) Feature extraction
2) Feature dimension reduction 
3) Stock maret prediction
higher dimension -> lower dimension with high variance
"""

from sklearn.datasets import load_iris

import pandas as pd 

iris = load_iris()

data = iris.data
feature_names = iris.feature_names
y = iris.target # target 

df = pd.DataFrame(data,columns = feature_names)
df["sinif"] = y # target

x = data 
#PCA
from sklearn.decomposition import PCA

pca = PCA(n_components = 2, whiten = True) # whiten->normalisation
pca.fit(x) # x-axis, decrease the dimension 

x_pca = pca.transform(x) # 4->2

print("variance ratio: ",pca.explained_variance_ratio_)

print("sum: ",sum(pca.explained_variance_ratio_)) # how many percent data we lost

#

df["p1"] = x_pca[:,0]
df["p2"] = x_pca[:,1]

color = ["red","green","blue"]

import matplotlib.pyplot as plt

for each in range(3):
    plt.scatter(df.p1[df.sinif == each],df.p2[df.sinif == each],color = color[each],label = iris.target_names[each])
    
plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()
    