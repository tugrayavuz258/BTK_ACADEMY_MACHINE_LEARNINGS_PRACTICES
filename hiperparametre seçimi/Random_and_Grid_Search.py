# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:01:24 2024

@author: TUGRA1
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier()
knn_param_grid = {'n_neighbors': np.arange(2, 31)}
knn_grid_search = GridSearchCV(knn, knn_param_grid)
knn_grid_search.fit(X_train, y_train)
print("KNN Grid Search Best Parameters:", knn_grid_search.best_params_)
print("KNN Grid Search Best Accuracy:", knn_grid_search.best_score_)

knn_random_search = RandomizedSearchCV(knn, knn_param_grid, n_iter=10)
knn_random_search.fit(X_train, y_train)
print("KNN Random Search Best Parameters:", knn_random_search.best_params_)
print("KNN Random Search Best Accuracy:", knn_random_search.best_score_)