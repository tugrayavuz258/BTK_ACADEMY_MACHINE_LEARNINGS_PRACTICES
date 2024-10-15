# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 03:21:30 2024

@author: TUGRA1
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# İris veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Veriyi eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı sınıflandırıcısı oluştur
tree = DecisionTreeClassifier()

# Hiperparametre uzayı
tree_param_dist = {'max_depth': [3, 5, 7]}

# K-Fold çapraz doğrulama
kf = KFold(n_splits=10)
tree_grid_search_kf = GridSearchCV(tree, tree_param_dist, cv=kf)
tree_grid_search_kf.fit(X_train, y_train)
print("K-Fold En iyi parametre:", tree_grid_search_kf.best_params_)
print("K-Fold En iyi acc:", tree_grid_search_kf.best_score_)

# Leave One Out çapraz doğrulama
loo = LeaveOneOut()
tree_grid_search_loo = GridSearchCV(tree, tree_param_dist, cv=loo)
tree_grid_search_loo.fit(X_train, y_train)
print("LOO En iyi parametre:", tree_grid_search_loo.best_params_)
print("LOO En iyi acc:", tree_grid_search_loo.best_score_)