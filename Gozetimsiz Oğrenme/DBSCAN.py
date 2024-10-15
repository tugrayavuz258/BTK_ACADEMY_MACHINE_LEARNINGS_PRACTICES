# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 01:01:46 2024

@author: TUGRA1
"""

from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Veri seti oluşturma
X, _ = make_circles(n_samples=1000, factor=0.5, noise=0.08, random_state=42)

# Veriyi görselleştirme
plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.title("Örnek Veri")
plt.show()

# DBSCAN modelini oluşturma ve eğitme
dbscan = DBSCAN(eps=0.15, min_samples=10)
cluster_labels = dbscan.fit_predict(X)

# Kümelenmiş verinin görselleştirilmesi
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis')
plt.title("DBSCAN Sonuçları")
plt.show()