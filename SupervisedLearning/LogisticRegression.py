import numpy as np
import matplotlib.pyplot as plt
import math, copy
from sklearn.linear_model import LogisticRegression



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost_logistic(x, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """
    m,n = x.shape
    cost=0.0
    for i in range(m):
        z=np.dot(x[i],w)+b
        f=sigmoid(z)
        cost+=-1*((y[i]*np.log(f))+((1-y[i])*(np.log(1-f))))
    cost = cost/m
    return cost

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
        z = np.dot(x[i],w) + b
        y_hat = sigmoid(z)
        diff = y_hat-y[i]
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
    J_history = []
    for i in range(num_iters):
        (dj_dw,dj_db) = gradient_function(x,y,w_in,b_in)
        w_in = w_in-alpha*dj_dw
        b_in = b_in-alpha*dj_db
        if i < 100000:  # prevent resource exhaustion
            J_history.append(compute_cost_logistic(x, y, w_in, b_in))
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    return w_in, b_in

def predict(x, w, b):
    m = x.shape[0]
    ans =np.zeros(m)
    for i in range(m):
        z = np.dot(x[i], w) + b
        ans[i] = sigmoid(z)
    return ans




x = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])
lr_model = LogisticRegression()
lr_model.fit(x,y)

y_pred = lr_model.predict(x)
print("Prediction on training set:", y_pred)
print("Accuracy on training set:", lr_model.score(x, y))


print("Manual Implementation")
m,n = x.shape
w_out, b_out = gradient_descent(x, y, np.zeros(n), 0.0, 0.01, 10000)

print(f"\nupdated parameters: w:{w_out}, b:{b_out}")
y_pred = predict(x, w_out, b_out)
print("Prediction on training set:", np.round(y_pred, 4))
