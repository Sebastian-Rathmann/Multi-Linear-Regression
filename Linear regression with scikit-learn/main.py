import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from functions import grad_dec, model_func


# Training matrix X, n(0)=Size in M^2 - n(1)=Number of years - n(2)= Distance from the center in Km.
X_train=np.array([[150.0,10.0,5.0], [200.0,15.0,10.0], [120.0,5.0,2.0], [180.0,12.0,7.0],[220.0,20.0,12.0],[250.0,8.0,3.0],[170.0,6.0,4.0],
                  [190.0,14.0,8.0], [210.0,18.0,11.0], [240.0,10.0,6.0], [260.0,7.0,5.0],[130.0,3.0,1.0],[200.0,11.0,9.0],[140.0,9.0,4.0],[180.0,16.0,7.0]])

#Vector training and, house prices on a scale of 1 to 1000.
y_train=np.array([[250.0],[300.0],[180.0],[280.0],[320.0],[270.0],[230.0],[290.0],[310.0],[260.0],[275.0],[160.0],[295.0],[240.0],[275.0]])

# Perform Scaling to improve the performance of the Gradient Decent
X_min=np.min(X_train, axis=0)
X_max=np.max(X_train, axis=0)
X_scaled = (X_train - X_min) / (X_max - X_min)

# Create a graph to compare the weight of each parameter with respect to the price
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes[0, 0].scatter(X_scaled[:, 0], y_train, marker='x')
axes[0, 0].set_title('Size')
axes[0, 1].scatter(X_scaled[:, 1], y_train, marker='x', c='y')
axes[0, 1].set_title('Number of years')
axes[1, 0].scatter(X_scaled[:, 2], y_train, marker='x', c='r')
axes[1, 0].set_title('Distance to the center')
axes[1, 1].axis('off')
plt.tight_layout()
plt.show()

# Initial parameters for Gradient Decent 
inicial_w = np.array([0, 0, 0])
inicial_b = 0
interactions = 7000
alpha = 0.0001

w, b, J_history = grad_dec.Gradient_Decent(X_scaled, y_train, alpha, interactions, inicial_w, inicial_b)

# Interactions to predict
prediction= np.array([0, 0, 0])
print("Home pricing prediction!")
prediction_temp = input("\nHow much size is your house in m^2? ")
prediction[0] = float(prediction_temp)
prediction_temp = input("\nHow old is your home? ")
prediction[1] = float(prediction_temp)
prediction_temp =input("\nHow far away are you from the center? ")
prediction[2] = float(prediction_temp)


# Scaling the input values of the prediction
prediction = (prediction - X_min) / (X_max - X_min)  

# Perform scaled prediction
y_hat = model_func.model(prediction, w, b)

print(f"The house is around U$D {int(y_hat[0]*1000)}")


# Cost Comparison Chart  
plt.plot(J_history)
plt.title("Cost vs. Iterations")
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()


# Graph to see the result of the linear regression
y_hat1 = model_func.normal_model(X_scaled[:, 0], w[0], b)
y_hat2 = model_func.normal_model(X_scaled[:, 1], w[1], b)
y_hat3 = model_func.normal_model(X_scaled[:, 2], w[2], b)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes[0, 0].scatter(X_scaled[:, 0], y_train, marker='x')
axes[0, 0].plot(X_scaled[:, 0], y_hat1)
axes[0, 0].set_title('Size')
axes[0, 1].scatter(X_scaled[:, 1], y_train, marker='x', c='y')
axes[0, 1].plot(X_scaled[:, 1], y_hat2)
axes[0, 1].set_title('Number of years')
axes[1, 0].scatter(X_scaled[:, 2], y_train, marker='x', c='r')
axes[1, 0].plot(X_scaled[:, 2], y_hat3) 
axes[1, 0].set_title('Distance to the center')
axes[1, 1].axis('off')
plt.tight_layout()
plt.show()