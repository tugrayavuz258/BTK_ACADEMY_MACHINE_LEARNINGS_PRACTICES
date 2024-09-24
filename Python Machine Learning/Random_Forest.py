# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:23:29 2024

@author: TUGRA1
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Veri setini yükle
faces = fetch_olivetti_faces()

# Veri setinden ilk iki yüzü görselleştir
fig, axes = plt.subplots(1, 2)
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.images[i], cmap='gray')
plt.show()

# Veriyi özellik matrisi (X) ve hedef değişken (y) olarak ayır
X = faces.data
y = faces.target

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rastgele orman sınıflandırıcısı oluştur ve eğit
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Test setinde tahmin yap
y_pred = rf.predict(X_test)

# Modelin doğruluğunu hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk:", accuracy)

#%%

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# California konut fiyatları verisini yükleme
california_housing = fetch_california_housing()

# Veriyi özelliklere (X) ve hedef değişkene (y) ayırma
X = california_housing.data
y = california_housing.target

# Veriyi eğitim ve test kümelerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rastgele Orman regresyon modeli oluşturma ve eğitme
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)

# Test kümesi üzerinde tahmin yapma
y_pred = rf_reg.predict(X_test)

# Ortalama karesel hata (MSE) hesaplama
mse = mean_squared_error(y_test, y_pred)

# Kök ortalama kare hatası (RMSE) hesaplama
rmse = np.sqrt(mse)

# Sonucu ekrana yazdırma
print("RMSE:", rmse)




