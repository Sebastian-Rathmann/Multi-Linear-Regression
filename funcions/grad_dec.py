#  Cost Funcion in multiple lineal regresiion:   𝐽(𝑤,𝑏) = 1/2𝑚 * ∑(𝑖=0 | 𝑚−1) (𝐰⋅𝐱(𝑖)+𝑏)− 𝑦(𝑖) )^2
#
#       𝑤𝑗 = 𝑤𝑗 − alpha * dJ/dw
# 
#       b = b − alpha * dJ/db
#
#       dJ/dw(j) = 1/𝑚 *∑(𝑖=0 | 𝑚−1) (𝐰⋅𝐱(𝑖)+𝑏) − 𝑦(𝑖))* 𝑥(𝑖)(j)
#
#       dJ/db = 1/𝑚 *∑(𝑖=0 | 𝑚−1) (𝐰⋅𝐱(𝑖)+𝑏) − 𝑦(𝑖))

import numpy as np

def cost_funcion(X, y, w, b):
    
    m=len(y)
    cost = 0
    for i in range (m):
        cost = cost + ((np.dot(w, X[i]) + b) - y[i])**2

    cost = cost / (2 * m)
    return cost

def Compute_Gradant(X, y, w, b):

    m, n=X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        temp = (np.dot(w , X[i]) + b) - y[i]
        dj_db = dj_db + temp
        for j in range (n):
            dj_dw[j] = dj_dw[j] + temp * X[i, j]
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def Gradient_Decent(X, y, alpha, num_interaccions, w, b):

    J_history = []

    for i in range(num_interaccions):

        dj_dw, dj_db = Compute_Gradant(X, y, w, b)

        w = w - alpha * dj_dw
 
        b = b - alpha * dj_db

        J_history.append(cost_funcion(X, y, w, b))
    
    return w, b, J_history

    