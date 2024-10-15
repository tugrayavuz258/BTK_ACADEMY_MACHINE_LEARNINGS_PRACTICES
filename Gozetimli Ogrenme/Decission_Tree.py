# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 03:41:03 2024

@author: TUGRA1
"""

# Gerekli kütüphaneleri içe aktar
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# İris veri setini yükle
iris = load_iris()

# Veri setini özellikler (X) ve hedef etiketler (y) olarak ayır
X = iris.data  # Özellikler (çanak yaprağı uzunluğu, çanak yaprağı genişliği, taç yaprağı uzunluğu, taç yaprağı genişliği)
y = iris.target  # Hedef etiket (iris çiçeğinin türü)

# Veri setini eğitim ve test setlerine ayır (verilerin %80'i eğitim, %20'si test için)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar Ağacı modelini oluştur ve eğit
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
tree_clf.fit(X_train, y_train)  # Modeli eğitim verileriyle eğit

# Modelin doğruluğunu test et
y_pred = tree_clf.predict(X_test)  # Test seti üzerinde tahminler yap
accuracy = accuracy_score(y_test, y_pred)  # Modelin doğruluğunu hesapla
print("İris veri seti ile eğitilen DT modeli doğruluğu: ", accuracy)

# Karışıklık matrisi (confusion matrix) oluştur ve yazdır
conf_matrix = confusion_matrix(y_test, y_pred)  # Karışıklık matrisini hesapla
print("conf_matrix: ")
print(conf_matrix)

# Karar ağacını görselleştir
plt.figure(figsize=(15,10))  # Görselleştirme boyutunu ayarla
plot_tree(tree_clf, filled=True, feature_names=iris.feature_names, class_names=list(iris.target_names))
plt.show()

# Karar ağacının karar verdiği noktada hangi özelliklerin ne kadar önemli olduğunu hesapla
feature_importances = tree_clf.feature_importances_  # Özelliklerin önem derecelerini al
feature_names = iris.feature_names  # Özelliklerin isimlerini al

# Özelliklerin önem derecelerini büyükten küçüğe doğru sırala
feature_importances_sorted = sorted(zip(feature_importances, feature_names), reverse=True)

# Özelliklerin isimlerini ve önem derecelerini yazdır
for importance, feature_name in feature_importances_sorted:
    print(f"{feature_name}: {importance}")


##for this dataset and model the sepal width feature is not important.

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay

# İris veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Sınıf sayısını ve renkleri belirle
n_classes = len(iris.target_names)
plot_colors = "ryb"

# Farklı özellik çiftleri için karar sınırlarını çizdir
for pairidx, pair in enumerate([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]):
    # Özellik çiftini seç
    X_pair = X[:, pair]

    # Karar ağacı sınıflandırıcısını oluştur ve eğit
    clf = DecisionTreeClassifier().fit(X_pair, y)

    # Alt grafik oluştur
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    # Karar sınırlarını çizdir
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X_pair,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]]
    )

    # Her bir sınıf için veri noktalarını farklı renkte çizdir
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X_pair[idx, 0], X_pair[idx, 1], c=color, label=iris.target_names[i], cmap=plt.cm.RdYlBu, edgecolors='black')
   
    plt.legend()

plt.show()
#%%
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Diyabet veri setini yükle
diabetes = load_diabetes()

# Özellikleri (X) ve hedef değişkeni (y) ayır
X = diabetes.data  # Özellikler
y = diabetes.target  # Hedef değişken (diyabet hastalığının ilerleme seviyesi)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı regresyon modeli oluştur ve eğit
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

# Test setinde tahmin yap
y_pred = tree_reg.predict(X_test)

# Ortalama kare hatası (MSE) hesapla
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

# Kök ortalama kare hatası (RMSE) hesapla
rmse = np.sqrt(mse)
print("RMSE:", rmse)













































   

