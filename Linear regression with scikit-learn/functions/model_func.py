import numpy as np

#𝑓(𝐱)=𝐰⋅𝐱+𝑏

def model(x, w, b):
    y_hat=np.dot(x,w) + b
    return y_hat

def normal_model(x, w , b):
    y_hat = w * x +b
    return y_hat