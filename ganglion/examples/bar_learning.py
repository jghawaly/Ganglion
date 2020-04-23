import sys
sys.path.append("../base")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, IFNeuralGroup, LIFNeuralGroup, FTLIFNeuralGroup, ExLIFNeuralGroup, AdExNeuralGroup, HSLIFNeuralGroup, AMLIFNeuralGroup, HSAMLIFNeuralGroup
from NeuralNetwork import NeuralNetwork
from NetworkRunHandler import NetworkRunHandler
from parameters import IFParams, LIFParams, FTLIFParams, ExLIFParams, AdExParams, PairSTDPParams, TripletSTDPParams, HSLIFParams, AMLIFParams, HSAMLIFParams, SynapseParams
from units import *
from utils import poisson_train, calculate_phi
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import argparse
from viz import VizWindow
import pyglet


def show_full_weight_map(group2, group1_label, group2_label, nn):
    nr = group2.field_shape[0]
    nc = group2.field_shape[1]

    fig, ax = plt.subplots(nr, nc)

    # get images and maximum pixel value
    imgs = []
    pix_max = 0
    for i in range(group2.n_num):
        img = nn.get_w_between_g_and_n(group1_label, group2_label, i)
        pix_max = max([pix_max, img.max()])
        imgs.append(img)

    # plot images
    i = 0
    for r in range(nr):
        for c in range(nc):
            img = ax[r, c].imshow(imgs[i], vmin=0, vmax=pix_max)
            ax[r, c].axis('off')
            i += 1

    fig.colorbar(img, ax=ax.ravel().tolist())
    plt.show()


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


def gen_bar_dataset(s):
    data = []
    labels = []
    for i in range(2):
        for j in range(s):
            img = np.zeros((s, s), dtype=np.float)
            if i == 0:  # vertical
                img[:, j] = 1.0
            else:  # horizontal
                img[j, :] = 1.0
            
            data.append(img / img.sum())
            labels.append(str(i))
    
    np.save('../datasets/bars/bar_data.npy', data)
    np.save('../datasets/bars/bar_labels.npy', labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation of a neural network that learns to detect bars in an image.')
    parser.add_argument('--model', type=str, default='amlif', help='the neuron model to evaluate, if, lif, ftlif, exlif, adex, hslif, amlif, or hsamlif')
    parser.add_argument('--increment', type=float, default=0.05, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--grid_size', type=int, default=8, help='length and width of square grid that bars are generated on')
    parser.add_argument('--stdp', type=str, default='pair', help='form of stdp to use, can be pair or triplet')
    parser.add_argument('--target_frequency', type=float, default=20, help='target frequency in Hz of neuron (only applicable to HSLIF neurons.')
    parser.add_argument('--e', type=int, default=160, help='number of episodes to run')

    args = parser.parse_args()
    # Define the time keeper and total duration of training
    tki = TimeKeeperIterator(timeunit=args.increment*msec)

    # select the requested neuron model and neuron model parameters
    if args.model == 'if':
        model = IFNeuralGroup
        e_params = IFParams()
        i_params = IFParams()
    elif args.model == 'lif':
        model = LIFNeuralGroup
        e_params = LIFParams()
        i_params = LIFParams()
    elif args.model == 'amlif':
        model = AMLIFNeuralGroup
        e_params = AMLIFParams()
        i_params = AMLIFParams()
    elif args.model == 'hsamlif':
        model = HSAMLIFNeuralGroup
        e_params = HSAMLIFParams()
        # set phi for homeostasis
        e_params.phi = calculate_phi(args.target_frequency, tki)
        i_params = e_params
    elif args.model == 'ftlif':
        model = FTLIFNeuralGroup
        e_params = FTLIFParams()
        i_params = FTLIFParams()
    elif args.model == 'exlif':
        model = ExLIFNeuralGroup
        e_params = ExLIFParams()
        i_params = ExLIFParams()
    elif args.model == 'adex':
        model = AdExNeuralGroup
        e_params = AdExParams()
        i_params = AdExParams()
    elif args.model == 'hslif':
        model = HSLIFNeuralGroup
        e_params = HSLIFParams()
        i_params = HSLIFParams()
        # set phi
        e_params.phi = calculate_phi(args.target_frequency, tki)
        i_params.phi = e_params.phi
    else:
        raise RuntimeError("%s is not a valid neuron model, must be if, lif, ftlif, exlif, adex, or hslif." % args.model)
    
    # select requested parameters for the type of learning to be used
    if args.stdp == 'pair':
        stdp_params = PairSTDPParams()
    elif args.stdp == 'triplet':
        stdp_params = TripletSTDPParams()
    else:
        raise RuntimeError("%s is not a valid stdp model, must be pair or triplet." % args.stdp)
  
    e_params.tao_m = 10.0 * msec
    e_params.gbar_e = 0.2 * nsiem
    e_params.gbar_i = 100 * nsiem

    s_params = SynapseParams()
    s_params.tao_syn_e = 0.5 * msec
    s_params.tao_syn_i = 0.5 * msec

    # define the neural groups
    g1 = SensoryNeuralGroup(1, args.grid_size * args.grid_size, "inputs", 1, tki, e_params, field_shape=(args.grid_size, args.grid_size), viz_layer_pos=(0,0))
    g2 = model(1, 36, "exc", 2, tki, e_params, field_shape=(6,6), viz_layer_pos=(0,0))

    # define the neural network
    nn = NeuralNetwork([g1, g2], "bar_learner", tki)

    # set learning parameters
    lp = stdp_params
    lp.lr = 0.001

    # connect inputs layer to the excitatory layer
    nn.fully_connect("inputs", "exc", trainable=True, stdp_params=lp, s_type=args.stdp, minw=0.05, maxw=0.1)
    # connect the excitatory layer to itself in the form of lateral and self-inhibition
    nn.fully_connect("exc", "exc", w_i=10.0, trainable=False, skip_one_to_one=False, s_type='inhib', syn_params=s_params)

    # load data
    training_data = np.load('../datasets/bars/bar_data.npy')
    training_labels = np.load('../datasets/bars/bar_labels.npy')
    network_labels = None
    
    # define order in which network will run
    run_order = ["inputs", "exc"]

    # Define the run handler, this handles the processing pipeline
    nrh = NetworkRunHandler(nn,
                            training_data,
                            training_labels,
                            network_labels,
                            run_order,
                            output_encoding=NetworkRunHandler.COUNT_ENC,
                            enable_additive_frequency=False,
                            enable_subtractive_frequency=False,
                            base_input_frequency=3000.0,
                            enable_noise_sampling=True,
                            noise_sampling_interval=3.0,
                            noise_probability=0.01,
                            normalize_on_weight_change=False,
                            normalize_on_start=False,
                            episodes=args.e,
                            save_at_end=False,
                            save_dir=None,
                            reset_on_process=False,
                            rewards=(-1.0, 1.0),
                            data_pre_processor=None,
                            training=True,
                            exposure_hits=1,
                            no_label=True,
                            encoding_count=2,
                            verbose=True)
    
    window = VizWindow(nrh, no_show_inhib=True)
    pyglet.app.run()
    nn.save_w('../models/bars')

    # show weights
    show_full_weight_map(g2, "inputs", "exc", nn)
