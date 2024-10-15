# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 00:29:36 2024

@author: TUGRA1
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Veri seti oluşturma
X, y = make_blobs(n_samples=300, centers=4, cluster_std=2, random_state=42)

# Veriyi görselleştirme
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Örnek Veri")
plt.show()

# K-Means modelini oluşturma ve eğitme
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Küme etiketlerini elde etme
labels = kmeans.labels_

# Kümelenmiş verileri ve küme merkezlerini görselleştirme
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
plt.title("K-Means")
plt.show()