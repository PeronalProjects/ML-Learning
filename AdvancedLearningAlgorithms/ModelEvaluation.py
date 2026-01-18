import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# reduce display precision on numpy arrays
np.set_printoptions(2)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
x=[1651.  ,
    1691.82,
    1732.63,
    1773.45,
    1814.27,
    1855.08,
    1895.9 ,
    1936.71,
    1977.53,
    2018.35,
    2059.16,
    2099.98,
    2140.8 ,
    2181.61,
    2222.43,
    2263.24,
    2304.06,
    2344.88,
    2385.69,
    2426.51,
    2467.33,
    2508.14,
    2548.96,
    2589.78,
    2630.59,
    2671.41,
    2712.22,
    2753.04,
    2793.86,
    2834.67,
    2875.49,
    2916.31,
    2957.12,
    2997.94,
    3038.76,
    3079.57,
    3120.39,
    3161.2 ,
    3202.02,
    3242.84,
    3283.65,
    3324.47,
    3365.29,
    3406.1 ,
    3446.92,
    3487.73,
    3528.55,
    3569.37,
    3610.18,
    3651.  ]
y=[432.65,454.94,471.53,482.51,468.36,482.15,540.02,534.58,558.35,566.42,581.4 ,596.46,596.71,619.45,616.58,653.16,666.52,670.59,669.02,678.91,707.44,710.76,745.19,729.85,743.8 ,738.2 ,772.95,772.22,784.21,776.43,804.78,833.27,825.69,821.05,833.82,833.06,825.7 ,843.58,869.4 ,851.5 ,863.18,853.01,877.16,863.74,874.67,877.74,874.11,882.8 ,910.83,897.42]

data = np.array([x,y]).T
x = data[:,0]
y = data[:,1]

# Convert 1-D arrays into 2-D
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

'''
Dividing the model into - 
training set - to train the model
cross-validation set - to evaluate different model configurations to choose from. example (feature engineering/which polynomial feature to be add to dataset)
test set - used to give a fair estimate of your chosen model's performance against new examples. Not used for decision making while training models
'''

'''60% training set'''
x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=.40, random_state=1)
'''20-20 test/cross-validation set'''
x_cv, x_test, y_cv, y_test = train_test_split(x_rest, y_rest, test_size=.50, random_state=1)
print(f"the shape of the training set (input) is: {x_train.shape}")
print(f"the shape of the training set (target) is: {y_train.shape}\n")
print(f"the shape of the cross validation set (input) is: {x_cv.shape}")
print(f"the shape of the cross validation set (target) is: {y_cv.shape}\n")
print(f"the shape of the test set (input) is: {x_test.shape}")
print(f"the shape of the test set (target) is: {y_test.shape}")

scaler = StandardScaler()
X_trained_scaled = scaler.fit_transform(x_train)

print(f"Computed mean of the training set: {scaler.mean_.squeeze():.2f}")
print(f"Computed standard deviation of the training set: {scaler.scale_.squeeze():.2f}")

linear_model = LinearRegression()
linear_model.fit(X_trained_scaled, y_train)

"""
Evaluating the model
"""

y_hat = linear_model.predict(X_trained_scaled)
print(f"training MSE (using sklearn function): {mean_squared_error(y_train, y_hat) / 2}")

X_cv_scaled = scaler.transform(x_cv)
y_cv_predict = linear_model.predict(X_cv_scaled)
print(f"training MSE for cross-validation (using sklearn function): {mean_squared_error(y_cv, y_cv_predict) / 2}")



# Instantiate the class to make polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_trained_mapped = poly.fit_transform(x_train)
print(X_trained_mapped[:5])
scaler_poly = StandardScaler()
X_trained_mapped_scaled = scaler_poly.fit_transform(X_trained_mapped)
print(X_trained_mapped_scaled[:5])
model = LinearRegression()
model.fit(X_trained_mapped_scaled, y_train)
y_hat = model.predict(X_trained_mapped_scaled)
print(f"training MSE (using sklearn function): {mean_squared_error(y_train, y_hat) / 2}")
x_cv_mapped = poly.transform(x_cv)
X_cv_mapped_scaled = scaler_poly.transform(x_cv_mapped)
y_cv_predict = model.predict(X_cv_mapped_scaled)
print(f"training MSE for cross-validation (using sklearn function): {mean_squared_error(y_cv, y_cv_predict) / 2}")
"""
The above process can be tried with different order polynomial values to get the lesser mean squared error models
"""


