from timekeeper import TimeKeeperIterator
from NeuralGroup import AdExNeuralGroup, SensoryNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import AdExParams, STDPParams, SynapseParams
from units import *
from utils import poisson_train
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import time
import sys

if __name__ == "__main__":
    # Load the data --------------------------------
    img_dir = "C:\\Users\\james\\CODE\\Ganglion\\ganglion\\data\\"
    # imgs = []
    edges = []

    on_c_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float)
    off_c_kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.float)
    for filename in os.listdir(img_dir):
        if filename.endswith(".JPEG"):
            img = cv2.imread(os.path.join(img_dir, filename), 0)
            img = cv2.resize(img, (64, 64), cv2.INTER_AREA)
            # imgs.append(img)

            # filter out edges
            edge = cv2.Canny(img, 100, 200)
            edge = cv2.resize(edge, (32,32), cv2.INTER_AREA)
            # cv2.imshow("", edge)
            # cv2.waitKey(0)
            edges.append(edge)
    # Define the network ---------------------------
    tki = TimeKeeperIterator(timeunit=0.5*msec)
    duration = 10000.0 * msec
    g1 = SensoryNeuralGroup(np.ones(1024, dtype=np.int), "inputs", tki, AdExParams(), field_shape=(32, 32))
    g2 = AdExNeuralGroup(np.ones(64, dtype=np.int), "exc", tki, AdExParams(), field_shape=(8,8))
    g3 = AdExNeuralGroup(np.zeros(64, dtype=np.int), "inh_lateral", tki, AdExParams(), field_shape=(8, 8))

    nn = NeuralNetwork([g1, g2, g3], "slugs", tki)
    lp = STDPParams()
    lp.lr = 0.0001
    sp = SynapseParams()
    sp.spike_window = 15.0 * msec
    # nn.fully_connect("inputs", "exc", trainable=True, stdp_params=lp, minw=0.01, maxw=0.4, syn_params=sp)
    nn.convolve_connect("inputs", "exc", np.zeros((4, 4), dtype=np.int), 4, 4, trainable=True, stdp_params=lp, minw=0.01, maxw=0.4, syn_params=sp)
    nn.one_to_one_connect("exc", "inh_lateral", w_i=1.0, trainable=False, syn_params=sp)
    nn.fully_connect("inh_lateral", "exc", w_i=1.0, trainable=False, skip_one_to_one=True, syn_params=sp)

    vms = []

    train_data = edges

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
    st = time.time()
    counter = 0
    d = train_data[counter]
    lts = 0
    skip = False
    for step in tki:
        if (step - lts)*tki.dt() >= 120*msec:
            counter += 1
            if counter == len(edges):
                counter = 0
            lts = step
            d = train_data[counter]
            skip = not skip

        if skip:
            g1.run(np.zeros(g1.field_shape, dtype=np.int))
        else:
            # inject spikes into sensory layer
            g1.run(poisson_train(d, tki.dt(), 5))
        # run all layers
        nn.run_order(["inputs", "exc", "inh_lateral"])
        
        sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

        if step >= duration/tki.dt():
            break

    print("\n\n")
    et = time.time()
    print(et-st)
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




