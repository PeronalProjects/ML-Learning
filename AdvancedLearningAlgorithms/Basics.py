import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import sympy
# from tensorflow.keras.layers import Dense, Input
# from tensorflow.keras import Sequential
# from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
# from tensorflow.keras.activations import sigmoid
# plt.style.use('./deeplearning.mplstyle')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

'''
TensorFlow operates on multidimensional arrays or tensors represented as tf.Tensor objects. Here is a two-dimensional tensor:
TensorFlow implements standard mathematical operations on tensors, as well as many operations specialized for machine learning. similar to numpy
'''
x = tf.constant([[1., 2., 3.],
                 [4., 5., 6.]])

print(x)
print(x.shape)
print(x.dtype)
'''
Note: Typically, anywhere a TensorFlow function expects a Tensor as input, 
the function will also accept anything that can be converted to a Tensor using tf.convert_to_tensor. See below for an example.
All tensors are immutable like Python numbers and strings: you can never update the contents of a tensor, only create a new one.
'''
print(tf.convert_to_tensor([1,2,3]))
print(tf.reduce_sum([1,2,3]))
if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

# This will be an int32 tensor by default; see "dtypes" below.
# Here is a "scalar" or "rank-0" tensor . A scalar contains a single value, and no "axes".
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

# A "matrix" or "rank-2" tensor has two axes:
# If you want to be specific, you can set the dtype (see below) at creation time
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)
print(rank_2_tensor)

'''
You can convert a tensor to a NumPy array either using np.array or the tensor.numpy method:
'''
print(np.array(rank_2_tensor))
print(rank_2_tensor.numpy())


'''
You can do basic math on tensors, including addition, element-wise multiplication, and matrix multiplication.
'''
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2], dtype=tf.int32)`

print(tf.add(a, b), "\n")
print(tf.multiply(a, b), "\n")
print(tf.matmul(a, b), "\n")

c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# Find the largest value
print(tf.reduce_max(c))
# Find the index of the largest value
print(tf.math.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))

rank_4_tensor = tf.zeros([3, 2, 4, 5])
print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())


'''
Variables
Normal tf.Tensor objects are immutable. To store model weights (or other mutable state) in TensorFlow use a tf.Variable.
'''
var  = tf.Variable([0.,0.,0.])
var.assign([1,2,3])
var.assign_add([1,1,1])
print(var)

'''
Automatic differentiation
Gradient descent and related algorithms are a cornerstone of modern machine learning.

To enable this, TensorFlow implements automatic differentiation (autodiff), which uses calculus to compute gradients. 
'''

x = tf.Variable(1.0)

def f(x):
  y = x**2 + 2*x - 5
  return y

with tf.GradientTape() as tape:
  y = f(x)

g_x = tape.gradient(y, x)  # g(x) = dy/dx

print(f(x))
print(g_x)


'''
Derivatives in python
'''
J,w = sympy.symbols('J,w')
J=(w**3)+(w**2)+w
print(J)
der = sympy.diff(J,w)
print(der)
ans = der.subs([(w,2)])
print(ans)
