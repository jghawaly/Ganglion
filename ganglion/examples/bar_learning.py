import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import FTLIFNeuralGroup, SensoryNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import FTLIFParams, STDPParams
from units import *
from utils import poisson_train
import numpy as np
import time
import random
import matplotlib.pyplot as plt


def form_full_weight_map(group2, group1_label, group2_label, nn):
    arr = []
    for n in range(group2.shape[0]):
        wmap = nn.get_w_between_g_and_n(group1_label, group2_label, n)
        arr.append(wmap)
    arr = np.array(arr)
    nr = group2.field_shape[0]
    nc = group2.field_shape[1]
    for row in range(nr):
        row_img = np.hstack(arr[row*nc:row*nc +nc, :, :])
        if row == 0:
            img = row_img
        else:
            img = np.vstack((img, row_img))

    return img


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
    tki = TimeKeeperIterator(timeunit=0.25*msec)
    duration = 100000.0 * msec

    inhib_layer_params = FTLIFParams()
    inhib_layer_params.gbar_i = 40.0 * nsiem
    inhib_layer_params.tao_m = 50 * msec

    exc_layer_params = FTLIFParams()
    exc_layer_params.gbar_e = 20.0 * nsiem
    exc_layer_params.tao_m = 50 * msec

    g1 = SensoryNeuralGroup(np.ones(64, dtype=np.int), "inputs", tki, exc_layer_params, field_shape=(8, 8))
    g2 = FTLIFNeuralGroup(np.ones(16, dtype=np.int), "exc", tki, exc_layer_params, field_shape=(4,4))
    g3 = FTLIFNeuralGroup(np.zeros(16, dtype=np.int), "inh_lateral", tki, inhib_layer_params, field_shape=(4,4))
    # g2.tracked_vars = ["spike"]

    nn = NeuralNetwork([g1, g2, g3], "bar_learner", tki)
    lp = STDPParams()
    # lp.lr_pre = 0.05
    # lp.lr_post = 0.05
    # lp.stdp_tao_pre = 20*msec
    nn.fully_connect("inputs", "exc", trainable=True, stdp_params=lp, minw=0.01, maxw=0.3)
    nn.one_to_one_connect("exc", "inh_lateral", w_i=1.0, trainable=False)
    nn.fully_connect("inh_lateral", "exc", w_i=1.0, trainable=False, skip_one_to_one=True)

    vms = []

    # -------------------------------------------------------
    img = form_full_weight_map(g2, "inputs", "exc", nn)
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
        if (step - lts)*tki.dt() >= 200*msec:
            lts = step
            d = genbar(8, 8)
            skip = not skip

        if skip:
            g1.run(np.zeros(g1.field_shape, dtype=np.int))
        else:
            # inject spikes into sensory layer
            g1.run(poisson_train(d, tki.dt(), 64))
        # run all layers
        nn.run_order(["inputs", "exc", "inh_lateral"])
        
        sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

        if step >= duration/tki.dt():
            break
    
    print("\n\n")

    # -------------------------------------------------------
    img = form_full_weight_map(g2, "inputs", "exc", nn)
    plt.imshow(img)
    plt.title("Post-training weight matrix")
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()
    # -------------------------------------------------------
    
    # v = [i[0] for i in g2.spike_track] 
    # times = np.arange(0,len(v), 1) * tki.dt() / msec

    # plt.plot(times, v)
    # plt.title("Voltage Track")
    # plt.xlabel("Time (msec)")
    # plt.ylabel("Membrane Potential (volt)")
    # plt.show()
