from timekeeper import TimeKeeperIterator
from NeuralGroup import AdExNeuralGroup, SensoryNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import AdExParams, STDPParams
from units import *
from utils import poisson_train, load_mnist
import numpy as np
import time
import sys
import random
import matplotlib.pyplot as plt


if __name__ == "__main__":
    start = time.time()
    tki = TimeKeeperIterator(timeunit=0.5*msec)
    duration = 10000.0 * msec
    g1 = SensoryNeuralGroup(np.ones(784, dtype=np.int), "inputs", tki, AdExParams(), field_shape=(28, 28))
    g2 = AdExNeuralGroup(np.ones(36, dtype=np.int), "exc", tki, AdExParams(), field_shape=(6,6))
    g3 = AdExNeuralGroup(np.zeros(36, dtype=np.int), "inh_lateral", tki, AdExParams(), field_shape=(6,6))
    g4 = AdExNeuralGroup(np.zeros(36, dtype=np.int), "inh_lateral2", tki, AdExParams(), field_shape=(6,6))
    # g3.tracked_vars = ["v_m"]

    nn = NeuralNetwork([g1, g2, g3, g4], "mnist", tki)
    lp = STDPParams()
    lp.lr = 0.001
    nn.fully_connect("inputs", "exc", trainable=True, stdp_params=lp, minw=0.01, maxw=0.4)
    nn.one_to_one_connect("exc", "inh_lateral", w_i=1.0, trainable=False)
    nn.one_to_one_connect("exc", "inh_lateral2", w_i=1.0, trainable=False)
    nn.fully_connect("inh_lateral", "exc", w_i=1.0, trainable=False, skip_one_to_one=True)
    nn.fully_connect("inh_lateral2", "exc", w_i=1.0, trainable=False, skip_one_to_one=True)

    vms = []

    # load MNIST
    train_data, train_labels, test_data, test_labels = load_mnist()

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
    mnist_counter = 0
    d = train_data[mnist_counter]
    lts = 0
    skip = False
    for step in tki:
        if (step - lts)*tki.dt() >= 120*msec:
            mnist_counter += 1
            lts = step
            d = train_data[mnist_counter]
            skip = not skip

        if skip:
            g1.run(np.zeros(g1.field_shape, dtype=np.int))
        else:
            # inject spikes into sensory layer
            g1.run(poisson_train(d, tki.dt(), 20))
        # run all layers
        nn.run_order(["inputs", "exc", "inh_lateral", "inh_lateral2"])
        
        sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

        if step >= duration/tki.dt():
            break
    
    print("\n\n")
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
