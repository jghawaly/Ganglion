import sys
sys.path.append("../base")

from timekeeper import TimeKeeperIterator
from units import *
from utils import StructuredMNIST, unit_normalize, log, apply_log_filter, max_normalize

import cv2
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import argparse
from numba import jit
from scipy.signal import convolve2d as convolve
import argparse


def log_filter(sigma, size=7):
    # generates a LOG filter for the given sigma and kernel size
    if size % 2 == 0:
        raise ValueError("LOG size must be odd, but is %s" % str(size))
    center = size // 2 
    indices = np.abs(np.arange(0, size, 1)-center)
    k = log(indices, 0, sigma)
    return k[np.newaxis]


class SaccadicEngine:
    def __init__(self, tki, horizontal_speed, vertical_speed, kernel_size=7, sigma=0.5, order=('+h', '+v')):
        # General ----------------------------------------------------------
        self.tki = tki

        # kernel properties ------------------------------------------------
        self.ksize = kernel_size
        self.sigma = sigma
        self.hfilter = log_filter(sigma, size=kernel_size)
        self.vfilter = log_filter(sigma, size=kernel_size).transpose()

        # saccade properties -----------------------------------------------
        self.order = order  # order of saccade motion
        # convert pixels per msec to msec per pixel
        self.hd = 1.0 / horizontal_speed 
        self.vd = 1.0 / vertical_speed
        # calculate number of cycles per saccade
        self.h_cycles = self.hd / tki.dt() * msec
        self.v_cycles = self.vd / tki.dt() * msec

        # check that the time discretization is sufficient for the requested sweeping speeds
        if self.h_cycles < 1:
            raise RuntimeError("Insufficient timeunit for horizontal saccadic input generation, either lower the horizontal sweep speed or the timeunit.")
        if self.v_cycles < 1:
            raise RuntimeError("Insufficient timeunit for vertical saccadic input generation, either lower the vertical sweep speed or the timeunit.")
        # convert to integers
        self.h_cycles = int(round(self.h_cycles))
        self.v_cycles = int(round(self.v_cycles))

    # NOTE: Should add a generator option here
    def generate(self, img):
        # this will contain the series of outputs
        series = []

        # run LOG filter in horizontal direction
        h_filtered = apply_log_filter(img, self.hfilter)
        h_filtered[h_filtered<0] = 0
        h_filtered = unit_normalize(h_filtered)
        # run LOG filter in vertical direction
        v_filtered = apply_log_filter(img, self.vfilter)
        v_filtered[v_filtered<0] = 0
        v_filtered = unit_normalize(v_filtered)

        # sorted indices for each direction
        hrange = range(img.shape[1])
        vrange = range(img.shape[0])

        for o in self.order:
            # select orientation and direction of saccade based on order
            r = hrange if o[1] == 'h' else vrange
            r = r if o[0] == '+' else r[::-1]

            # select cycle based on orientation
            numcycles = self.h_cycles if o[1] == 'h' else self.v_cycles

            # iterate over image
            for i in r:
                next_img = np.zeros_like(img)
                if o[1] == 'h':
                    next_img[:, i] = h_filtered[:, i].copy() 
                else:
                    next_img[i, :] = v_filtered[i, :].copy()
                # append this selection for as many cycles as required
                for c in range(numcycles):
                    series.append(next_img)
        
        return np.array(series)


# this and the following function is for plotting the gif
imgs = []
def get(p):
    for txt in fig.texts:
        txt.set_visible(False)
    tii, sii = imgs[p]
    for m in ax: 
        m.cla()
        m.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                      labelbottom=False, labeltop=False, labelleft=False, labelright=False) 
    ax[0].imshow(tii, cmap='gray')
    plt.title("Original", fontsize=14)
    ax[1].imshow(sii, cmap='gray')
    plt.title("Events", fontsize=14)
    fig.text(0.5, 0.04, 'Time: %.2f msec' % (p * tki.dt() / msec), va='center', ha='center', fontsize=14)
    fig.set_tight_layout(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run saccadic event-based input encoding example on the MNIST dataset.')
    parser.add_argument('--save', action='store_true', help='save the gifs to the current working directory')
    parser.add_argument('--noshow', action='store_true', help='disables the display of the outputs')
    parser.add_argument('--speed', type=float, default=1.4, help="saccadic speed in pixels per millisecond")
    parser.add_argument('--sigma', type=float, default=0.5, help="Gaussian sigma value")
    parser.add_argument('--k', type=int, default=7, help="kernel size, must be ODD")
    args = parser.parse_args()

    ii = 0
    f = log_filter(args.sigma, size=args.k)
    plt.imshow(f)
    plt.colorbar()
    plt.title("Filter")
    plt.show()
    
    tki = TimeKeeperIterator(timeunit=0.2*msec)

    s_engine = SaccadicEngine(tki, args.speed, args.speed, sigma=args.sigma, kernel_size=args.k)

    smnist = StructuredMNIST((0,1,2,3,4,5,6,7,8,9))
    train, label = smnist.all()
    i=0
    for t in train:
        fig, ax = plt.subplots(ncols=2, sharex=True)

        t = max_normalize(t)
        saccades = s_engine.generate(t)
        for s in saccades:
            s = max_normalize(s)
            imgs.append((t, s))

        # generate gif
        animator = FuncAnimation(fig, get, frames=len(imgs), interval=len(imgs) * tki.dt() * 10)
        if args.save:
            animator.save('digit%s_%s.gif' % (label[i], i), writer=PillowWriter(fps=20))

        if not args.noshow:
            plt.show()
        else:
            fig.clf()
        
        imgs = []
        i += 1
        if i == 10:
            exit()

