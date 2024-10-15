# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 21:17:11 2024

@author: TUGRA1
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# Iris veri seti yükleniyor
iris = load_iris()
X = iris.data
y = iris.target

# Veri seti eğitim ve test olarak bölünüyor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modeli tanımlanıyor
tree = DecisionTreeClassifier()

# Hiperparametreler için grid aralığı
tree_param_grid = {
    "max_depth": [3, 5, 7],
    "max_leaf_nodes": [None, 5, 10, 20, 30, 50]
}

# Çapraz doğrulama kat sayısı
nb_cv = 3

# GridSearchCV kullanılarak model eğitiliyor
tree_grid_search = GridSearchCV(tree, tree_param_grid, cv=nb_cv)
tree_grid_search.fit(X_train, y_train)

# En iyi parametrelerin ve doğruluğun yazdırılması
print("DT Grid Search Best Parameters:", tree_grid_search.best_params_)
print("DT Grid Search Best Accuracy: ", tree_grid_search.best_score_)

# Ortalama test skoru ve parametrelerin yazdırılması
for mean_score, params in zip(tree_grid_search.cv_results_["mean_test_score"], tree_grid_search.cv_results_["params"]):
    print(f"Ortalama test skoru: {mean_score}, Parametreler: {params}")

# Çapraz doğrulama sonuçlarının her katman (fold) için detaylı yazdırılması
cv_result = tree_grid_search.cv_results_
for i, params in enumerate(cv_result["params"]):
    print(f"Parametreler: {params}")
    for j in range(nb_cv):
        accuracy = cv_result[f"split{j}_test_score"][i]
        print(f"\tFold {j+1} - Accuracy: {accuracy}")
