import matplotlib.pyplot as plt
import numpy as np
from funcions import grad_dec, model_func

# Matriz entrenamiento X, P1=Tamaño en M^2  - P2=Cantidad de años - P3=Lejania al centro en Km.
X_train=np.array([[150.0,10.0,5.0], [200.0,15.0,10.0], [120.0,5.0,2.0], [180.0,12.0,7.0],[220.0,20.0,12.0],[250.0,8.0,3.0],[170.0,6.0,4.0],
                  [190.0,14.0,8.0], [210.0,18.0,11.0], [240.0,10.0,6.0], [260.0,7.0,5.0],[130.0,3.0,1.0],[200.0,11.0,9.0],[140.0,9.0,4.0],[180.0,16.0,7.0]])

#Vector entrenamiento y, precio de las viviendas en escala 1 a 1000.
y_train=np.array([[250.0],[300.0],[180.0],[280.0],[320.0],[270.0],[230.0],[290.0],[310.0],[260.0],[275.0],[160.0],[295.0],[240.0],[275.0]])


# Realizamos un Escalado para mejorar el rendimiendo del Gradiant Decent
m, n = X_train.shape
X_scaled = np.zeros_like(X_train)
y_scaled = np.zeros_like(y_train)

for i in range(m):
    y_scaled[i] = ((y_train[i] - np.min(y_train)) / (np.max(y_train) - np.min(y_train)))
    for j in range(n):
        X_scaled[i, j] = (X_train[i, j] - np.min(X_train)) / (np.max(X_train) - np.min(X_train))


# Realizamos un grafico para comprar el peso de cada parametro con respecto al precio
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes[0, 0].scatter(X_scaled[:, 0], y_scaled, marker='x')
axes[0, 0].set_title('Tamaño')
axes[0, 1].scatter(X_scaled[:, 1], y_scaled, marker='x', c='y')
axes[0, 1].set_title('Cantidad de años')
axes[1, 0].scatter(X_scaled[:, 2], y_scaled, marker='x', c='r')
axes[1, 0].set_title('Lejania al centro')
axes[1, 1].axis('off')
plt.tight_layout()
plt.show()


# Parametro iniciales para Gradient Decent
inicial_w = np.array([0, 0, 0])
inicial_b = 0
interations = 550
alpha = 0.003

w, b, J_history = grad_dec.Gradient_Decent(X_scaled, y_scaled, alpha, interations, inicial_w, inicial_b)

# Interacciones para predecir
prediccion= np.array([0, 0, 0])
print("Prediccion de precios de casa!")
prediccion_temp = input("\nDe cuanto tamaño es su casa en M^2? ")
prediccion[0] = float(prediccion_temp)
prediccion_temp = input("\nCuantos años tiene su casa? ")
prediccion[1] = float(prediccion_temp)
prediccion_temp =input("\nA cuantos km esta del centro? ")
prediccion[2] = float(prediccion_temp)


# Escalado de los valores de entrada de la predicción
prediccion = (prediccion - np.min(X_train)) / (np.max(X_train) - np.min(X_train))

# Realizar la predicción escalada
y_hat_scaled_predi = model_func.model(prediccion, w, b)

# Deshacer el escalado para obtener la predicción final
y_hat = y_hat_scaled_predi * (np.max(y_train) - np.min(y_train)) + np.min(y_train)

print(f"La casa está alrededor de U$D {int(y_hat[0]*1000)}")


# Grafico comparativo de Cost  
plt.plot(J_history)
plt.title("Costo vs. Iteraciones")
plt.xlabel('Iteración')
plt.ylabel('Costo')
plt.show()


# Grafico para ver el resultado del lineal regression
y_hat_scaled1 = model_func.normal_model(X_scaled[:, 0], w[0], b)
y_hat_scaled2 = model_func.normal_model(X_scaled[:, 1], w[1], b)
y_hat_scaled3 = model_func.normal_model(X_scaled[:, 2], w[2], b)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes[0, 0].scatter(X_scaled[:, 0], y_scaled, marker='x')
axes[0, 0].plot(X_scaled[:, 0], y_hat_scaled1)
axes[0, 0].set_title('Tamaño')
axes[0, 1].scatter(X_scaled[:, 1], y_scaled, marker='x', c='y')
axes[0, 1].plot(X_scaled[:, 1], y_hat_scaled2)
axes[0, 1].set_title('Cantidad de años')
axes[1, 0].scatter(X_scaled[:, 2], y_scaled, marker='x', c='r')
axes[1, 0].plot(X_scaled[:, 2], y_hat_scaled3) 
axes[1, 0].set_title('Lejania al centro')
axes[1, 1].axis('off')
plt.tight_layout()
plt.show()