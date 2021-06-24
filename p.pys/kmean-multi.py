#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#data for n_clusters=3
#x=np.array([12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72])
#y=np.array([39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24])
x=np.array([185,170,168,179,182,188])
y=np.array([72,56,60,68,72,77])
df = np.array(list(zip(x, y)))
plt.scatter(x,y)
kmeans=KMeans(n_clusters=2)
kmeans.fit(df)
c=kmeans.predict(df)
centroids=kmeans.cluster_centers_
print(centroids)
plt.scatter(centroids[:,0],centroids[:,1],c='r')
plt.show()

