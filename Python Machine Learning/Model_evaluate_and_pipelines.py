# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 04:33:44 2024

@author: TUGRA1
"""

from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np

# Sınıflandırma verisi oluştur
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
X += 1.2 * np.random.uniform(size=X.shape)
Xy = (X, y)

# Ay şeklinde veri oluştur
X, y = make_moons(noise=0.2, random_state=42)
moons = (X, y)

# Çember şeklinde veri oluştur
X, y = make_circles(noise=0.1, factor=0.3, random_state=42)
circles = (X, y)

# Tüm veri kümelerini bir listede topla
datasets = [Xy, moons, circles]

# Kullanılacak sınıflandırıcılar
names = ["SVC", "Decision Tree", "KNN"]
classifiers = [
    make_pipeline(StandardScaler(), SVC(kernel="rbf", gamma=2, C=1)),
    DecisionTreeClassifier(max_depth=5),
    KNeighborsClassifier(3)
]

# Veri kümelerini ve karar sınırlarını görselleştir
fig, axes = plt.subplots(len(datasets), len(classifiers) + 1, figsize=(10, 10))
i = 0

for ds_cnt, ds in enumerate(datasets):
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    ax = axes[ds_cnt, 0]
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors="black")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker='x', edgecolors="black")
    ax.set_title(f"Dataset {ds_cnt + 1}")

    for name, clf in zip(names, classifiers):
        ax = axes[ds_cnt, i + 1]
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(clf, X, cmap=plt.cm.coolwarm, alpha=0.7, ax=ax)
        
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors="black")
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, marker='x', edgecolors="black")
        
        ax.set_title(f"{name} (Acc: {score:.2f})")
        i += 1

    i = 0

plt.tight_layout()
plt.show()
