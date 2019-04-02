import numpy as np
import time
from units import *
from numba import njit, roc

# x = 0.5*np.ones((2,2), dtype=np.float)
# y = 0.1*np.ones((1,2), dtype=np.float)

# m = x.shape[0] * x.shape[1]
# n = y.shape[0] * y.shape[1]

# w = np.ones((m, n), dtype=np.float)

# print(x)
# print(y)
# print(w)
# print(np.dot(x.ravel(), w))
def fast_row_roll(val, assignment):
    """
    A JIT compiled method for roll the values of all rows in a givenm matrix down by one, and
    assigning the first row to the given assignment
    """
    val[1:,:] = val[0:-1,:]
    val[0] = assignment
    return val
def construct_dt_matrix(v):
    """
    Construct the matrix that relates the rows of self.history to the elapsed time. This should
    only be called once on initialization
    """
    delta_t = np.zeros(v.shape, dtype=float)
    times = np.arange(1, 5+1, 1) * 0.1
    for idx, val in np.ndenumerate(times):
        delta_t[idx, :] = val

    return delta_t
x = np.ones((5, 2, 2))
y = np.array([0, 0])

r = np.zeros((2,2))
r[np.where(y>0), :] = y[np.where(y>0)]
x[0] = r
print(x)