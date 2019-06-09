import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, IFNeuralGroup, LIFNeuralGroup, FTLIFNeuralGroup, ExLIFNeuralGroup, AdExNeuralGroup, HSLIFNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import IFParams, LIFParams, FTLIFParams, ExLIFParams, AdExParams, PairSTDPParams, TripletSTDPParams, HSLIFParams
from units import *
from utils import poisson_train, calculate_phi
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
    
    return i / i.sum()


def genbars(s1, s2):
    i = np.zeros((s1, s2), dtype=np.float)
    for _ in range(2):
        i[:, np.random.randint(0, s2)] = 1.0
    
    for _ in range(2):
        i[np.random.randint(0, s1), :] = 1.0
    
    return i / i.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation of a neural network that learns to detect bars in an image.')
    parser.add_argument('--model', type=str, default='hslif', help='the neuron model to evaluate, if, lif, ftlif, exlif, adex, or hslif')
    parser.add_argument('--duration', type=float, default=100000.0, help='duration of simulation (milliseconds)')
    parser.add_argument('--increment', type=float, default=0.5, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--exposure', type=float, default=40.0, help='the duration of time that the network is exposed to each training example')
    parser.add_argument('--input_rate', type=float, default=64.0, help='maximum firing rate of input sensory neurons (Hz)')
    parser.add_argument('--grid_size', type=int, default=16, help='length and width of square grid that bars are generated on')
    parser.add_argument('--stdp', type=str, default='pair', help='form of stdp to use, can be pair or triplet')
    parser.add_argument('--target_frequency', type=float, default=20, help='target frequency in Hz of neuron (only applicable to HSLIF neurons.')

    args = parser.parse_args()
    tki = TimeKeeperIterator(timeunit=args.increment*msec)
    duration = args.duration * msec

    if args.model == 'if':
        model = IFNeuralGroup
        params = IFParams()
    elif args.model == 'lif':
        model = LIFNeuralGroup
        params = LIFParams()
    elif args.model == 'ftlif':
        model = FTLIFNeuralGroup
        params = FTLIFParams()
    elif args.model == 'exlif':
        model = ExLIFNeuralGroup
        params = ExLIFParams()
    elif args.model == 'adex':
        model = AdExNeuralGroup
        params = AdExParams()
    elif args.model == 'hslif':
        model = HSLIFNeuralGroup
        params = HSLIFParams()
        params.phi = calculate_phi(args.target_frequency, tki)
    else:
        raise RuntimeError("%s is not a valid neuron model, must be if, lif, ftlif, exlif, adex, or hslif." % args.model)
    
    if args.stdp == 'pair':
        stdp_params = PairSTDPParams
    elif args.stdp == 'triplet':
        stdp_params = TripletSTDPParams
    else:
        raise RuntimeError("%s is not a valid stdp model, must be pair or triplet." % args.stdp)

    start = time.time()

    # exc_layer_params.gbar_e = 10.0 * nsiem
    # inhib_layer_params.gbar_i = 20.0 * nsiem
    # exc_layer_params.tao_m = 40 * msec

    g1 = SensoryNeuralGroup(1, args.grid_size * args.grid_size, "inputs", tki, params, field_shape=(args.grid_size, args.grid_size))
    g2 = model(1, 16, "exc", tki, params, field_shape=(4,4))
    g3 = model(0, 16, "inh_lateral", tki, params, field_shape=(4,4))

    nn = NeuralNetwork([g1, g2, g3], "bar_learner", tki)
    lp = stdp_params()
    lp.lr = 0.05

    nn.fully_connect("inputs", "exc", trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type=args.stdp)
    nn.one_to_one_connect("exc", "inh_lateral", w_i=1.0, trainable=False)
    nn.fully_connect("inh_lateral", "exc", w_i=1.0, trainable=False, skip_one_to_one=True)

    d = genbar(args.grid_size, args.grid_size)
    last_exposure_step = 0
    cummulative_spikes = 0
    f_add = 0
    for step in tki:
        if (step - last_exposure_step) * tki.dt() >= args.exposure * msec:
            if cummulative_spikes == 0:
                f_add += 5
            else:
                cummulative_spikes = 0
                last_exposure_step = step
                d = genbar(args.grid_size, args.grid_size)
                nn.reset()
                # nn.normalize_weights()
         
        # inject spikes into sensory layer
        g1.run(poisson_train(d, tki.dt(), args.input_rate + f_add))

        # run all layers
        nn.run_order(["inputs", "exc", "inh_lateral"])

        cummulative_spikes += g2.spike_count.sum()
        
        sys.stdout.write("Current simulation time :: %.0f ms :: Num Spikes :: %g                  \r" % (step * tki.dt() / msec, cummulative_spikes))

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
