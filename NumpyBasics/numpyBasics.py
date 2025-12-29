import numpy
import numpy as np

"""
In python, lists are slow
Numpy stands for numerical Python
Aims to provide array object 50X faster than python lists
Array object is called ndarray, it provides a lot of supporting functions that make working with ndarray very easy
"""

"""
Why faster?
Most NumPy arrays are contiguous in memory (all values stored next to each other)
"""

print(f"numpy versions : {np.__version__}")

"""
Print a basic numpy array
"""
arr = numpy.array([1,2,3,4,5])
print(arr)

print("\nDifferent array dimensions:")
a = np.array(42)
b = np.array([1, 2, 3, 4, 5])
c = np.array([[1, 2, 3], [4, 5, 6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])

print(f"{a.ndim} dimensional array : \n{a}\n")
print(f"{b.ndim} dimensional array : \n{b}\n")
print(f"{c.ndim} dimensional array : \n{c}\n")
print(f"{d.ndim} dimensional array : \n{d}\n")

print(f"adding elements 2 and 3 of 1D array : 3+4={b[2]+b[3]}")
print('2nd element on 1st row in 2D array: ', c[0, 1])

arr = np.array([1, 2, 3, 4, 5, 6, 7])

print(f"slicing array : {arr} from 1 till 4th index element : {arr[1:5]}")
print(f"slicing array : {arr} from 5 till last index element : {arr[5:]}")
print(f"slicing array : {arr} from 1 till 4th index element with steps of 2 : {arr[1:5:2]}")

zeroes = np.zeros(5)
ones = np.ones(5)
print(zeroes)
print(ones)
print(f"sum : np.sum(ones) : {np.sum(ones)}")
print(f"sum : np.mean(ones) : {np.mean(ones)}")
