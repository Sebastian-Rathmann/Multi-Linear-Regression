import matplotlib.pyplot as plt
import numpy as np
from functions import grad_dec, model_func
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler

# Training matrix X, n(0)=Size in M^2 - n(1)=Number of years - n(2)= Distance from the center in Km.
X_train=np.array([[150.0,10.0,5.0], [200.0,15.0,10.0], [120.0,5.0,2.0], [180.0,12.0,7.0],[220.0,20.0,12.0],[250.0,8.0,3.0],[170.0,6.0,4.0],
                  [190.0,14.0,8.0], [210.0,18.0,11.0], [240.0,10.0,6.0], [260.0,7.0,5.0],[130.0,3.0,1.0],[200.0,11.0,9.0],[140.0,9.0,4.0],[180.0,16.0,7.0]])

#Vector training and, house prices on a scale of 1 to 1000.
y_train=np.array([[250.0],[300.0],[180.0],[280.0],[320.0],[270.0],[230.0],[290.0],[310.0],[260.0],[275.0],[160.0],[295.0],[240.0],[275.0]])
y_train = y_train.ravel()

# Perform Scaling to improve the performance of the Gradient Decent.
X_min=np.min(X_train, axis=0)
X_max=np.max(X_train, axis=0)
X_scaled = (X_train - X_min) / (X_max - X_min)

# To confirm the scaling, I compare it with one made by Sklearn
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(X_train)

if np.allclose(data_scaled, X_scaled):
    print('=== Scale made correctly ===')
else:
    print('Not correct: ')
    print(f"\nData scaled: {data_scaled}")
    print(f"\nX_scaled: {X_scaled}")
    
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

# Initial parameters for Gradient Decent = = = Better values I found: interactions=20k - alpha=0.001 (predi= 234.018)
inicial_w = np.array([0, 0, 0])
inicial_b = 0
interactions = 5000
alpha = 0.001  
w, b, J_history = grad_dec.Gradient_Decent(X_scaled, y_train, alpha, interactions, inicial_w, inicial_b)

# Gradient Decent by Sklearn
Gradient_Sklearn = SGDRegressor(max_iter=5000)
Gradient_Sklearn.fit(X_scaled, y_train)
b_sklearn = Gradient_Sklearn.intercept_
w_sklearn = Gradient_Sklearn.coef_

# I compare both 'W' and 'B'
print(f"\nSklearn parameters:                   w: {w_sklearn}, b:{b_sklearn}\nMy parameters:                        w: {w}, b:{b}")

# Cost Comparison Chart  
plt.plot(J_history)
plt.title("Cost vs. Iterations")
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# Interactions to predict
prediction= np.array([0, 0, 0])
print("\nHome pricing prediction!")
prediction_temp = input("\nHow much size is your house in m^2? ")
prediction[0] = float(prediction_temp)
prediction_temp = input("\nHow old is your home? ")
prediction[1] = float(prediction_temp)
prediction_temp =input("\nHow far away are you from the center? ")
prediction[2] = float(prediction_temp)


# Scaling the input values of the prediction and then I check that it is done correctly
prediction_scaled = (prediction - X_min) / (X_max - X_min)
prediction_check = scaler.transform(prediction.reshape(1, -1))

rounded_prediction_scaled = np.round(prediction_scaled, decimals=10)
rounded_prediction_check = np.round(prediction_check, decimals=10)

if np.allclose(rounded_prediction_scaled, rounded_prediction_check): 
    print('=== Scale prediction made correctly ===')
else:
    print('Not correct: ')
    print(f"\nPrediction Scaled: {prediction_scaled}")
    print(f"\nX_scaled: {prediction_check}")

# Perform my prediction
y_hat = model_func.model(prediction_scaled, w, b)
print(f"\nMy house price prediction is around U$D {int(y_hat*1000)}")

# Perform Sklearn prediction
Sklearn_prediction_core= Gradient_Sklearn.predict(prediction_check)
Sklearn_prediction = int(Sklearn_prediction_core[0])*1000
print(f"Sklearn house price prediction is around U$D {Sklearn_prediction}")