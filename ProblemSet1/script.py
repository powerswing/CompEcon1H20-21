#%%
import numpy as np

# Exercise 1

# a) MATLAB backslash operator on python using numpy
A = [[2, 0, 1], [0, 4, 1], [1, -1, 4]]
b = [30, 40, 15]

def directMethod(A, b):
    """
    Solves the linear system directly by inversion.

    Arguments:
    A - coefficient matrix
    b - ordinate vector

    Return:
    x - solution vector
    """
    return np.dot(np.linalg.inv(A), b)

x = directMethod(A, b)

print('Solution vector x: {}'.format(x))
print('Check if x is correct: {}'.format(np.round(np.dot(A, x), 0) == b))

print('\n\nCoefficient matrix inversion is a direct method.')
# %%
