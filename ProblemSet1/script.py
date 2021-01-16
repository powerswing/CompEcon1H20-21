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

    Returns:
    x - solution vector
    """
    return np.dot(np.linalg.inv(A), b)

x = directMethod(A, b)

print('Solution vector x: {}'.format(x))
print('Check if x is correct: {}'.format(np.round(np.dot(A, x), 0) == b))

print('\n\nCoefficient matrix inversion is a direct method. We may use this method since the invertible of matrix A exists. Once this is not the case the solution to the system may be approximated with the help of iterative methods')
# %%
# b) Solving the linear system via Gauss-Seidel iteration algorithm
def gaussSeidel(A, b, maxIter=10e3, tol=1/10e3):
    """
    Solves the linear system via Gauss-Seidel iterative method. The initial guess is
    take as a null-vector of length of ordinate vector

    Arguments:
    A - coefficient matrix
    b - ardinate vector

    Returns:
    x - convergence vector
    """
    # check diagonal dominance of A
    for i in range (len(b)):
        diag = A[i][i]
        nondiag = 0
        for j in range(len(b)):
            nondiag += A[i][j] # sum of row elements

        if 2 * diag - nondiag < 0: # diag is multiplied by two since nondiag includes diag
            print('Matrix A is not strictly diagonally dominant at row: {}'.format(i))
    
    # start iteration
    iteration = 0
    iterCond = 0
    error = 1
    errorCond = 0
    x0 = [0.0 for i in range(len(b))]
    L = np.tril(A) # lower triangular matrix
    U = A - L # upper triangular matrix

    while iterCond or errorCond != 1:
        x1 = np.dot(np.linalg.inv(L), b - np.dot(U, x0))
        error = np.linalg.norm(x1-x0)
        iteration += 1
        x0 = x1

        if error < tol:
            errorCond = 1
        elif iterCond > maxIter:
            iterCond = 1
    
    print('Vector x {} converged at iteration {}'.format(x0, iteration))

print('\n\n Since the matrix is strictly diagonally dominant we may use the Gauss-Seidel method. The coefficient matrix is of shape (3,3) hence the method converged relatively fast. Due to very low error tolerance of 1/10e3 convergence vector x is very close to solution vector x found with the direct method of inversion of coefficient matrix')
# %%
# c) Computing condition number
def conditionNumber(A):
    return np.linalg.det(np.linalg.inv(A)) * np.linalg.det(A)