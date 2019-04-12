from timekeeper import TimeKeeperIterator
from NeuralGroup import AdExNeuralGroup, SensoryNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import AdExParams, STDPParams
from units import *
from utils import poisson_train
import numpy as np
import time
import sys
import random
import matplotlib.pyplot as plt


def genbar(s1, s2):
    i = np.zeros((s1, s2), dtype=np.float)
    if random.random() > 0.5:
        i[:, np.random.randint(0, s2)] = 1.0
    else:
        i[np.random.randint(0, s1), :] = 1.0
    
    return i


def genbars(s1, s2):
    i = np.zeros((s1, s2), dtype=np.float)
    for _ in range(2):
        i[:, np.random.randint(0, s2)] = 1.0
    
    for _ in range(2):
        i[np.random.randint(0, s1), :] = 1.0
    
    return i


if __name__ == "__main__":
    start = time.time()
    tki = TimeKeeperIterator(timeunit=0.5*msec)
    duration = 200000.0 * msec
    g1 = SensoryNeuralGroup(np.ones(64, dtype=np.int), "inputs", tki, AdExParams(), field_shape=(8, 8))
    g2 = AdExNeuralGroup(np.ones(32, dtype=np.int), "exc", tki, AdExParams(), field_shape=(4,8))
    g3 = AdExNeuralGroup(np.zeros(32, dtype=np.int), "inh_lateral", tki, AdExParams(), field_shape=(4,8))
    g4 = AdExNeuralGroup(np.zeros(32, dtype=np.int), "inh_lateral2", tki, AdExParams(), field_shape=(4,8))
    # g3.tracked_vars = ["v_m"]

    nn = NeuralNetwork([g1, g2, g3, g4], "bar_learner", tki)
    lp = STDPParams()
    lp.lr = 0.0001
    nn.fully_connect("inputs", "exc", trainable=True, stdp_params=lp, minw=0.01, maxw=0.4)
    nn.one_to_one_connect("exc", "inh_lateral", w_i=1.0, trainable=False)
    nn.one_to_one_connect("exc", "inh_lateral2", w_i=1.0, trainable=False)
    nn.fully_connect("inh_lateral", "exc", w_i=1.0, trainable=False, skip_one_to_one=True)
    nn.fully_connect("inh_lateral2", "exc", w_i=1.0, trainable=False, skip_one_to_one=True)

    vms = []

    # -------------------------------------------------------
    arr = []
    for n in range(g2.shape[0]):
        wmap = nn.get_w_between_g_and_n("inputs", "exc", n)
        arr.append(wmap)
    arr = np.array(arr)
    nr = g2.field_shape[0]
    nc = g2.field_shape[1]
    for row in range(nr):
        row_img = np.hstack(arr[row*nc:row*nc +nc, :, :])
        if row == 0:
            img = row_img
        else:
            img = np.vstack((img, row_img))

    plt.imshow(img)
    plt.title("Pre-training weight matrix")
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()

    # -------------------------------------------------------
    d = genbar(8, 8)
    lts = 0
    skip = False
    for step in tki:
        if (step - lts)*tki.dt() >= 50*msec:
            lts = step
            d = genbar(8, 8)
            skip = not skip

        if skip:
            g1.run(np.zeros(g1.field_shape, dtype=np.int))
        else:
            # inject spikes into sensory layer
            g1.run(poisson_train(d, tki.dt(), 200))
        # run all layers
        nn.run_order(["inputs", "exc", "inh_lateral", "inh_lateral2"])
        
        sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

        if step >= duration/tki.dt():
            break
    
    print("\n\n")
    # plt.imshow(nn.get_w_between_g_and_n("inputs", "exc", 0))
    # plt.title("Post-training Weight Matrix")
    # plt.colorbar()
    # plt.clim(0, 1)
    # plt.show()
    # -------------------------------------------------------
    arr = []
    for n in range(g2.shape[0]):
        wmap = nn.get_w_between_g_and_n("inputs", "exc", n)
        arr.append(wmap)
    arr = np.array(arr)
    nr = g2.field_shape[0]
    nc = g2.field_shape[1]
    for row in range(nr):
        row_img = np.hstack(arr[row*nc:row*nc +nc, :, :])
        if row == 0:
            img = row_img
        else:
            img = np.vstack((img, row_img))

    plt.imshow(img)
    plt.title("Pre-training weight matrix")
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()
    # -------------------------------------------------------
