import numpy as np
import time
from units import *
from numba import njit, roc

dt = 0.1 * msec
window = 1.0 * msec
n_synapses = 6
num = int(window/dt)

h = np.zeros((num, n_synapses), dtype=np.float)
w = 0.5*np.ones((1,n_synapses))

def roll_and_assign(val, assignment):
    val = np.roll(val, 1, axis=0)
    val[0] = assignment
    return val

@njit
def roll_and_assign2(val, assignment):
    val[1:,:] = val[0:-1,:]
    val[0] = assignment
    return val


f1 = np.array([1., 0., 0., 0., 1., 2.])

h = roll_and_assign2(h, f1)
print(h)

time.sleep(1)
f2 = np.array([0., 0., 1., 0., 0., 1.])
h = roll_and_assign2(h, f2)

print(h)
print()

time.sleep(1)

st = time.perf_counter()
for i in range(10000):
    h = roll_and_assign2(h, np.zeros_like(f2))
    # print(h)
    # time.sleep(1)

print(time.perf_counter() - st)