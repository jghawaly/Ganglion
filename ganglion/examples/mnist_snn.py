import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import ExLIFNeuralGroup, SensoryNeuralGroup, AdExNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import ExLIFParams, STDPParams, AdExParams
from units import *
from utils import poisson_train, load_mnist, save_img
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


if __name__ == "__main__":
    # Load MNIST data
    train_data, train_labels, test_data, test_labels = load_mnist()

    area = np.sum(train_data, axis=(1,2))
    max_area = area.max()


    start = time.time()
    tki = TimeKeeperIterator(timeunit=0.5*msec)
    duration = 200000.0 * msec

    inhib_layer_params = AdExParams()
    inhib_layer_params.gbar_i = 65.0 * nsiem
    inhib_layer_params.tao_m = 100 * msec

    exc_layer_params = AdExParams()
    exc_layer_params.gbar_e = 50.0 * nsiem
    exc_layer_params.tao_m = 100 * msec

    g1 = SensoryNeuralGroup(np.ones(784, dtype=np.int), "inputs", tki, exc_layer_params, field_shape=(28, 28))
    g2 = AdExNeuralGroup(np.ones(100, dtype=np.int), "exc", tki, exc_layer_params, field_shape=(10,10))
    g3 = AdExNeuralGroup(np.zeros(100, dtype=np.int), "inh_lateral", tki, inhib_layer_params, field_shape=(10,10))
    g3.tracked_vars = ["spike"]

    nn = NeuralNetwork([g1, g2, g3], "mnist_learner", tki)
    lp = STDPParams()
    lp.lr = 0.001
    lp.a2_minus = 8.0e-3
    lp.a3_minus = 3e-4
    nn.fully_connect("inputs", "exc", trainable=True, stdp_params=lp, minw=0.05, maxw=0.9)
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

    d = train_data[0]
    a = area[0]
    lts = 0
    skip = False
    mnist_counter = 0
    num_spikes = 1
    freq_add = 0
    for step in tki:
        if (step - lts)*tki.dt() >= 100*msec:
            num_spikes = np.sum(g3.spike_track)
            print(num_spikes)
            if num_spikes < 5:
                freq_add += 32
                lts = step
                print("testing new frequency: %g"%(64 + freq_add))
            else:
                freq_add = 0
                lts = step
                d = train_data[mnist_counter]
                a = area[mnist_counter]
                mnist_counter += 1
                skip = not skip
                save_img("C:/Users/james/CODE/junk/%s.jpg"%str(mnist_counter), form_full_weight_map(g2, "inputs", "exc", nn), normalize=True)
            
        if skip:
            g1.run(np.zeros(g1.field_shape, dtype=np.int))
        else:
            # inject spikes into sensory layer
            g1.run(poisson_train(d, tki.dt(), (64 + freq_add) * max_area / a))
        # run all layers
        nn.run_order(["inputs", "exc", "inh_lateral"])
        nn.normalize_weights()
        
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