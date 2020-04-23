import sys
sys.path.append("../base")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, IFNeuralGroup, LIFNeuralGroup, FTLIFNeuralGroup, ExLIFNeuralGroup, AdExNeuralGroup, HSLIFNeuralGroup, AMLIFNeuralGroup
from NeuralNetwork import NeuralNetwork
from NetworkRunHandler import NetworkRunHandler
from parameters import IFParams, LIFParams, FTLIFParams, ExLIFParams, AdExParams, PairSTDPParams, TripletSTDPParams, HSLIFParams, AMLIFParams, DASTDPParams, SynapseParams
from units import *
from utils import poisson_train, calculate_phi
import collections
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
            if nc!= 1:
                img = ax[r, c].imshow(imgs[i], vmin=0, vmax=pix_max)
                ax[r, c].axis('off')
            else:
                img = ax[r].imshow(imgs[i], vmin=0, vmax=pix_max)
                ax[r].axis('off')
            i += 1

    fig.colorbar(img, ax=ax.ravel().tolist())
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation of a neural network that learns to detect bars in an image.')
    parser.add_argument('--model', type=str, default='amlif', help='the neuron model to evaluate, if, lif, ftlif, exlif, adex, or hslif')
    parser.add_argument('--increment', type=float, default=0.05, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--grid_size', type=int, default=8, help='length and width of square grid that bars are generated on')
    parser.add_argument('--stdp', type=str, default='pair', help='form of stdp to use, can be pair or triplet')
    parser.add_argument('--target_frequency', type=float, default=20, help='target frequency in Hz of neuron (only applicable to HSLIF neurons.')
    parser.add_argument('--e', type=int, default=60, help='number of episodes to run')
    parser.add_argument('--viz', action='store_true', help='view network in GUI')
    parser.add_argument('--m', type=int, default=16, help='number of past rewards/accuracies to keep track of for averaging')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    args = parser.parse_args()
    # Define the time keeper and total duration of training
    tki = TimeKeeperIterator(timeunit=args.increment*msec)

    # select the requested neuron model and neuron model parameters
    if args.model == 'if':
        model = IFNeuralGroup
        input_params = IFParams()
        f_params = IFParams()
        h_params = IFParams()
        o_params = IFParams()
    elif args.model == 'lif':
        model = LIFNeuralGroup
        input_params = LIFParams()
        f_params = LIFParams()
        h_params = LIFParams()
        o_params = LIFParams()
    elif args.model == 'amlif':
        model = AMLIFNeuralGroup
        input_params = AMLIFParams()
        f_params = AMLIFParams()
        h_params = AMLIFParams()
        o_params = AMLIFParams()
    elif args.model == 'ftlif':
        model = FTLIFNeuralGroup
        input_params = FTLIFParams()
        f_params = FTLIFParams()
        h_params = FTLIFParams()
        o_params = FTLIFParams()
    elif args.model == 'exlif':
        model = ExLIFNeuralGroup
        input_params = ExLIFParams()
        f_params = ExLIFParams()
        h_params = ExLIFParams()
        o_params = ExLIFParams()
    elif args.model == 'adex':
        model = AdExNeuralGroup
        input_params = AdExParams()
        f_params = AdExParams()
        h_params = AdExParams()
        o_params = AdExParams()
    elif args.model == 'hslif':
        model = HSLIFNeuralGroup
        input_params = HSLIFParams()
        f_params = HSLIFParams()
        h_params = HSLIFParams()
        o_params = HSLIFParams()
        # set phi
        input_params.phi = calculate_phi(args.target_frequency, tki)
        f_params.phi = input_params.phi
        h_params.phi = input_params.phi
        o_params.phi = input_params.phi
    else:
        raise RuntimeError("%s is not a valid neuron model, must be if, lif, ftlif, exlif, adex, or hslif." % args.model)
    
    # input neuron group parameters
    input_params.tao_m = 10.0 * msec
    input_params.gbar_e = 1.8 * nsiem
    input_params.gbar_i = 120 * nsiem

    # feature neuron group parameters
    f_params.tao_m = 10.0 * msec
    f_params.gbar_e = 25 * nsiem

    # hidden neuron group parameters
    h_params.tao_m = 10.0 * msec
    h_params.gbar_e = 25 * nsiem

    # output neuron group parameters
    o_params.tao_m = 75 * msec

    # define the neural groups
    g1 = SensoryNeuralGroup(1, args.grid_size * args.grid_size, "inputs", 1, tki, input_params, field_shape=(args.grid_size, args.grid_size), viz_layer_pos=(0,0))
    g2 = model(1, 36, "features", 2, tki, f_params, field_shape=(6,6), viz_layer_pos=(0,0))
    g3 = model(1, 9, "hidden", 3, tki, h_params, field_shape=(3,3), viz_layer_pos=(0,0))
    g4 = model(1, 2, 'outputs', 4, tki, input_params, field_shape=(2, 1), viz_layer_pos=(0,0))

    # define the neural network
    nn = NeuralNetwork([g1, g2, g3, g4], "bar_learner", tki)

    # set learning parameters
    lp = DASTDPParams()
    lp.lr = args.lr
    lp.wd=0.95
    # time constants of eligbility traces
    lp.ab_et_tao = 10 * msec   
    lp.ba_et_tao = 10 * msec

    # scaling factors for positive rewards
    lp.ab_scale_pos = 1.0
    lp.ba_scale_pos = -1.0

    # scaling factors for negative rewards
    lp.ab_scale_neg = -1.0
    lp.ba_scale_neg = -1.0

    # synapse parameters
    s_params = SynapseParams()
    s_params.tao_syn_e = 0.5 * msec
    s_params.tao_syn_i = 0.5 * msec

    # connect inputs layer to the features layer
    nn.fully_connect("inputs", "features", trainable=False, s_type='base')
    # connect the features layer to itself in the form of lateral and self-inhibition
    nn.fully_connect("features", "features", w_i=20.0, trainable=False, skip_one_to_one=False, s_type='inhib', syn_params=s_params)
    # connect features layer to hidden layer
    nn.fully_connect("features", "hidden", trainable=True, stdp_params=lp, s_type='da', minw=0.1, maxw=0.3)
    # connect the hidden layer to itself in the form of lateral and self-inhibition
    nn.fully_connect("hidden", "hidden", w_i=10.0, trainable=False, skip_one_to_one=False, s_type='inhib', syn_params=s_params)
    # connect hidden layer to output layer
    nn.fully_connect("hidden", "outputs", trainable=True, stdp_params=lp, s_type='da', minw=0.1, maxw=0.3)

    # load data
    training_data = np.load('../datasets/bars/bar_data.npy')
    training_labels = np.load('../datasets/bars/bar_labels.npy')
    network_labels = np.array(['0', '1'])
    
    # define order in which network will run
    run_order = ["inputs", "features", "hidden", "outputs"]

    # load weights learned using stdp
    nn.load_w('../models/bars')

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
                            normalize_on_weight_change=True,
                            normalize_on_start=True,
                            episodes=args.e,
                            save_at_end=False,
                            save_dir=None,
                            reset_on_process=False,
                            rewards=(-1.0, 1.0),
                            training=True,
                            encoding_count=1,
                            output_is_ftm=False,
                            ftm_supervised=False)

    if args.viz:
        window = VizWindow(nrh, no_show_inhib=True)
        pyglet.app.run()
    else:
        num_metrics = 0
        running = True
        # these will store reward and accuracy
        cum_reward = []  # for plotting
        cum_accuracy = []  # for plotting
        r_deque = collections.deque(maxlen=args.m)  # rolling window
        a_deque = collections.deque(maxlen=args.m)  # rolling window
        try:
            eps = []
            while running:
                running, metrics = nrh.run_episode()
                r, a, cs = metrics

                r_deque.append(r)
                a_deque.append(a)

                num_metrics += 1
                if num_metrics >= args.m:
                    eps.append(nrh.current_episode)
                    cr = np.mean(r_deque)
                    ca = np.mean(a_deque)
                    cum_reward.append(cr)
                    cum_accuracy.append(ca)
                    print("Episode %s ::  Last Reward  ::  %s  Average Reward %.2f :: Average Accuracy %.2f :: Cummulative Spike Count %s" % (nrh.current_episode-1, r, cr, ca, str(cs)))
                else:
                    print("Building reward and accuracy history")
            np.save('../models/bars/eps_%s_twolayer.npy' % str(args.lr), eps)
            np.save('../models/bars/acc_%s_twolayer.npy' % str(args.lr), cum_accuracy)
            plt.plot(eps, cum_accuracy, color='k', linestyle='-', marker='.')
            plt.xlabel("Episode")
            plt.ylabel("Average Accuracy (N=30)")
            plt.show()
        except KeyboardInterrupt:
            exit()

