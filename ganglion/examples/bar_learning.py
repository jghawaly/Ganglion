import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, IFNeuralGroup, LIFNeuralGroup, FTLIFNeuralGroup, ExLIFNeuralGroup, AdExNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import IFParams, LIFParams, FTLIFParams, ExLIFParams, AdExParams, STDPParams
from units import *
from utils import poisson_train
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import argparse


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
    parser = argparse.ArgumentParser(description='Run a simulation of a single neuron.')
    parser.add_argument('model', type=str, help='the neuron model to evaluate')
    parser.add_argument('--duration', type=float, default=500000.0, help='duration of simulation (milliseconds)')
    parser.add_argument('--increment', type=float, default=0.2, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--exposure', type=float, default=200.0, help='the duration of time that the network is exposed to each training example')
    parser.add_argument('--input_rate', type=float, default=64.0, help='maximum firing rate of input sensory neurons (Hz)')
    parser.add_argument('--grid_size', type=int, default=16, help='length and width of square grid that bars are generated on')
    parser.add_argument('--stdp', type=str, default='pair', help='form of stdp to use, can be pair or triplet')

    args = parser.parse_args()

    if args.model == 'if':
        model = IFNeuralGroup
        params = IFParams
    elif args.model == 'lif':
        model = LIFNeuralGroup
        params = LIFParams
    elif args.model == 'ftlif':
        model = FTLIFNeuralGroup
        params = FTLIFParams
    elif args.model == 'exlif':
        model = ExLIFNeuralGroup
        params = ExLIFParams
    elif args.model == 'adex':
        model = AdExNeuralGroup
        params = AdExParams
    else:
        raise RuntimeError("%s is not a valid neuron model, must be if, lif, ftlif, exlif, or adex.")

    start = time.time()
    tki = TimeKeeperIterator(timeunit=args.increment*msec)
    duration = args.duration * msec

    inhib_layer_params = LIFParams()
    inhib_layer_params.gbar_i = 150.0 * nsiem
    inhib_layer_params.tao_m = 100 * msec

    exc_layer_params = params()
    exc_layer_params.gbar_e = 100.0 * nsiem
    exc_layer_params.tao_m = 100 * msec

    g1 = SensoryNeuralGroup(np.ones(args.grid_size * args.grid_size, dtype=np.int), "inputs", tki, exc_layer_params, field_shape=(args.grid_size, args.grid_size))
    g2 = model(np.ones(16, dtype=np.int), "exc", tki, exc_layer_params, field_shape=(4,4))
    g3 = LIFNeuralGroup(np.zeros(16, dtype=np.int), "inh_lateral", tki, inhib_layer_params, field_shape=(4,4))

    nn = NeuralNetwork([g1, g2, g3], "bar_learner", tki)
    lp = STDPParams()

    nn.fully_connect("inputs", "exc", trainable=True, stdp_params=lp, minw=0.01, maxw=0.3, stdp_form=args.stdp)
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

    d = genbar(args.grid_size, args.grid_size)
    lts = 0
    skip = False
    for step in tki:
        if (step - lts) * tki.dt() >= args.exposure * msec:
            lts = step
            d = genbar(args.grid_size, args.grid_size)
            skip = not skip

        if skip:
            g1.run(np.zeros(g1.field_shape, dtype=np.int))
        else:
            # inject spikes into sensory layer
            g1.run(poisson_train(d, tki.dt(), args.input_rate))
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
