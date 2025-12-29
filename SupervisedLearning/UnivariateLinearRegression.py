import numpy as np
import matplotlib.pyplot as plt
import math, copy
from sklearn.linear_model import LinearRegression


x_train = np.array([1.0, 2.0])           #(size in 1000 square feet)
y_train = np.array([300.0, 500.0])           #(price in 1000s of dollars)

"""
Note : for Uni variate linear regression, there would be single w & b.
"""
def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
      x (ndarray (m,)): Data, m examples
      y (ndarray (m,)): target values
      w,b (scalar)    : model parameters

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    m = x.shape[0]
    sum = 0
    for i in range(m):
        y_hat = w*x[i]+b
        error = (y_hat-y[i])**2
        sum+=error
    return sum/(2*m)

def gradient_function(x,y,w,b):
    """
        Computes the gradient for linear regression
        Args:
          x (ndarray (m,)): Data, m examples
          y (ndarray (m,)): target values
          w,b (scalar)    : model parameters
        Returns
          dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
          dj_db (scalar): The gradient of the cost w.r.t. the parameter b
    """
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        y_hat = (w * x[i]) + b
        dj_dw+=(y_hat-y[i])*x[i]
        dj_db+=(y_hat-y[i])
    dj_dw = dj_dw/(m)
    dj_db = dj_db/(m)
    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Performs gradient descent to fit w,b. Updates w,b by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x (ndarray (m,))  : Data, m examples
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters
      alpha (float):     Learning rate
      num_iters (int):   number of iterations to run gradient descent
      cost_function:     function to call to produce cost
      gradient_function: function to call to produce gradient

    Returns:
      w (scalar): Updated value of parameter after running gradient descent
      b (scalar): Updated value of parameter after running gradient descent
      J_history (List): History of cost values
      p_history (list): History of parameters [w,b]
      """
    J_history = []
    p_history = []
    for i in range(num_iters):
        (dj_dw,dj_db) = gradient_function(x,y,w_in,b_in)
        w = w_in-alpha*dj_dw
        b = b_in-alpha*dj_db
        w_in = w
        b_in = b
        if i < 100000:  # prevent resource exhaustion
            J_history.append(compute_cost(x, y, w_in, b_in))
            p_history.append([w_in, b_in])
            # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w_in: 0.3e}, b:{b_in: 0.5e}")
    return w_in, b_in, J_history, p_history

x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 4, 5, 6, 7])
def predict(input):
    (w, b, J_history, p_history) = gradient_descent(x, y, 0.0, 0.0, 0.05, 1000)
    target = w*input+b
    return target

print(predict(9))

## Using Scikit Library
x=x.reshape(-1,1)
reg = LinearRegression().fit(x, y)
print(reg.predict(np.array([[9]])))






