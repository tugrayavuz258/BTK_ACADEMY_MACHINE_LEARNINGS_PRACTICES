# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:03:34 2024

@author: TUGRA1
"""

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris() 
X = iris.data 
y = iris.target

pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X)

plt.figure()
for i in range(len(iris.target_names)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label = iris.target_names[i])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Iris Dataset")
plt.legend()
