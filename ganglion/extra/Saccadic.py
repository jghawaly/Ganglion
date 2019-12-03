import sys
sys.path.append("../base")

from timekeeper import TimeKeeperIterator
from units import *
from utils import StructuredMNIST

import cv2
import numpy as np 
import matplotlib.pyplot as plt
import argparse
from numba import jit
from scipy.signal import convolve2d as convolve


@jit(nopython=True, parallel=True, nogil=True)
def log1D(d, sigma):
    # calculates the value of a LOG filter at the given index, d and sigma
    return -1/(np.pi*np.power(sigma, 4.0)) * (1-(d**2)/(2* sigma**2))*np.exp(-(d**2)/(2* sigma**2))


def log_filter(sigma, size=7):
    # generates a LOG filter for the given sigma and kernel size
    if size % 2 == 0:
        raise ValueError("LOG size must be odd, but is %s" % str(size))
    center = size // 2 
    indices = np.abs(np.arange(0, size, 1)-center)
    k = np.array([log1D(indices, sigma)])

    return k


def apply_log_filter(img, f):
    newimg = convolve(img, f, mode='same', boundary='fill')
    return newimg / newimg.max()


class SaccadicEngine:
    def __init__(self, left_sweep_speed=1.4, right_sweep_speed=1.4, kernel_size=7, sigma=0.5, order=('lr', 'du')):
        # kernel properties ------------------------------------------------
        self.ksize = kernel_size
        self.sigma = sigma
        self.hfilter = log_filter(sigma, size=kernel_size)
        self.vfilter = log_filter(sigma, size=kernel_size).transpose()

        # saccade properties -----------------------------------------------
        self.order = order  # order of saccade motion
        # convert pixels per msec to msec per pixel
        self.ld = 1.0 / left_sweep_speed  
        self.rd = 1.0 / right_sweep_speed  

    


if __name__ == "__main__":
    f = log_filter(0.5, size=7)
    plt.imshow(f)
    plt.show()

    smnist = StructuredMNIST((0,1,2,3,4,5,6,7,8,9))
    train, label = smnist.all()
    i=0
    for t in train:
        o = t / t.max()
        t = apply_log_filter(t, f)
        t[t<0] = 0
        print(o.shape)
        print(t.shape)
        plt.imshow(np.hstack((o, t)), cmap='gray')
        plt.show()
        i+=1
        if i==5:
            exit()

