# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 05:42:14 2024

@author: TUGRA1
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Rastgele veri oluşturma
np.random.seed(42)  # Sonuçları tekrarlanabilir yapmak için seed belirledik
X = np.random.rand(100, 1)
y = 3 + 4 * X + np.random.randn(100, 1)  # y = 3 + 4x + noise

# Doğrusal regresyon modeli oluşturma ve eğitme
model = LinearRegression()
model.fit(X, y)

# Tahminler yapma
y_pred = model.predict(X)

# Verileri ve regresyon doğrusunu çizdirme
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Gerçek Veri")
plt.plot(X, y_pred, color='red', label="Regresyon Doğrusu")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Doğrusal Regresyon")
plt.legend()
plt.grid(True)
plt.show()

# Model parametrelerini yazdırma0
print("Katsayı (a1):", model.coef_[0][0])
print("Sabit terim (a):", model.intercept_[0])

#%%
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
# Diyabet verisini yükleme
diabetes = load_diabetes()
diabetes_X, diabetes_y = load_diabetes(return_X_y=True)

# Veriyi eğitim ve test setlerine ayırma
diabetes_X = diabetes_X[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Doğrusal regresyon modeli oluşturma ve eğitme
lin_reg = LinearRegression()
lin_reg.fit(diabetes_X_train, diabetes_y_train)

# Test setinde tahmin yapma
diabetes_y_pred = lin_reg.predict(diabetes_X_test)

# Modelin performansını değerlendirme
mse = mean_squared_error(diabetes_y_test, diabetes_y_pred)
r2 = r2_score(diabetes_y_test, diabetes_y_pred)
print("mse: ", mse)
print("r2: ", r2)

# Sonuçları görselleştirme
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue")
plt.show()