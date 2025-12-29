import numpy as np
import matplotlib.pyplot as plt
import math, copy
from sklearn.linear_model import LinearRegression


x_train = np.array([1.0, 2.0])           #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])           #(price in 1000s of dollars)

"""
Note : for MultiVariate linear regression, there would be vector w & single b.
"""


def compute_cost(x, y, w, b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """
    m = x.shape[0]
    sum = 0
    for i in range(m):
        y_hat = np.dot(x[i], w)+b
        error = (y_hat-y[i])**2
        sum+=error
    return sum/(2*m)


def gradient_function(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """
    m,n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        y_hat = np.dot(x[i],w) + b
        diff = (y_hat-y[i])
        for j in range(n):
            dj_dw[j]+=diff*x[i][j]
        dj_db+=(y_hat-y[i])
    dj_dw = dj_dw/(m)
    dj_db = dj_db/(m)
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent

    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      """
    for i in range(num_iters):
        (dj_dw,dj_db) = gradient_function(x,y,w_in,b_in)
        w = w_in-alpha*dj_dw
        b = b_in-alpha*dj_db
        w_in = w
        b_in = b
    return w_in, b_in

x = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
x_scaled = x / np.max(x, axis=0)  # simple min-max scaling

y = np.array([460, 232, 178])

def predict(input):
    m,n = x.shape
    (w, b) = gradient_descent(x_scaled, y, np.zeros(n), 0.0, 0.02, 10000)
    y_hat = np.dot(input/np.max(x, axis=0), w)+b
    return y_hat

print(predict(x[0, :]))

## Using Scikit Library
reg = LinearRegression().fit(x, y)
print(reg.predict(x[0, :].reshape(1,-1)))






