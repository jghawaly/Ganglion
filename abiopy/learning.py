import numpy as np
from units import *
from numba import jit

@jit(nopython=True)
def dw(dt, w, tao_plus, tao_minus, lr_plus, lr_minus, a_plus=0.3, a_minus=-0.6):
    """
    Calculate the STDP weight change in a single synapse
    dt: postsynaptic neuron spike time - presynaptic neuron spike time
    w: current synaptic weight
    tao_plus: STDP post-after-pre time constant
    tao_minus: STDP post-before-pre time constant
    lr_plus: STDP post-after-pre learning rate
    lr_minus: STDP post-before-pre learning rate
    """

    if dt >= 0:
        return lr_plus * (1.0 - w) * a_plus * np.exp(-dt / tao_plus)
    else:
        return lr_minus * w * a_minus * np.exp(dt / tao_minus)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    x = np.arange(-50 * msec, 50 * msec, 0.1 * msec)
    y = [dw(val, 0.5, 10 * msec, 10 * msec, 1.0, 1.0) for val in x]

    plt.plot(x, y)
    plt.title("STDP: w = 0.5")
    plt.xlabel("dt: Post-Pre")
    plt.ylabel("dw")
    plt.show()