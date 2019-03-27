import numpy as np
import time
from units import *

dt = 0.1 * msec
window = 1.0 * msec
n_synapses = 6
num = int(window/dt)

h = np.zeros((num, n_synapses), dtype=np.float)
w = 0.5*np.ones((1,n_synapses))

def construct_delta_t(hist_arr, num_hist):
    delta_t = np.zeros_like(hist_arr)
    times = np.arange(1, num_hist+1, 1) * dt
    for idx, val in np.ndenumerate(times):
        delta_t[idx, :] = val
    
    return delta_t

delta_t = construct_delta_t(h, num)

def roll_and_assign(val, assignment):
    val = np.roll(val, 1, axis=0)
    val[0] = assignment
    return val

def calc_isyn(h, we, delt, gbar=100*nsiem, tao_syn=5*msec, v_m=-75*mvolt, vrev=0*mvolt):
    return np.sum(h * we * (v_m - vrev) * gbar * np.exp(-delt / tao_syn))

f1 = np.array([1., 0., 0., 0., 1., 2.])

h = roll_and_assign(h, f1)
print(h)
print(delta_t)
print()
isyn = calc_isyn(h, w, delta_t)
print(isyn)
# print(np.sum(isyn))
exit()
time.sleep(1)
f2 = np.array([0., 0., 1., 0., 0., 1.])
h = roll_and_assign(h, f2)

print(h)
print()
time.sleep(1)

for i in range(10):
    h = roll_and_assign(h, np.zeros_like(f2))
    print(h)
    print(h*delta_t)
    print()
    time.sleep(5)