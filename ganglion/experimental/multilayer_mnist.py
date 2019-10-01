import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, IFNeuralGroup, LIFNeuralGroup, FTLIFNeuralGroup, ExLIFNeuralGroup, AdExNeuralGroup, HSLIFNeuralGroup
from NeuralNetwork import NeuralNetwork
from NetworkRunHandler import NetworkRunHandler
from parameters import IFParams, LIFParams, FTLIFParams, ExLIFParams, AdExParams, HSLIFParams, DASTDPParams
from units import *
from utils import poisson_train, calculate_phi, load_mnist, add_noise
from viz import VizWindow
import pyglet

import cv2
import numpy as np 
import numpy.random as nprand
import matplotlib.pyplot as plt
import argparse
import random
from collections import deque
import os


def sample_patch(img, rows, cols):
    random_row = nprand.randint(rows, img.shape[0])
    # row_indices = [random_row-rows, random_row+1]

    random_col = nprand.randint(cols, img.shape[1])
    # col_indices = [random_col-cols, random_col+1]

    return img[random_row-rows:random_row, random_col-cols:random_col] / img.max()


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
   
def resize(img):
    return cv2.resize(img, (16,16), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation of a neural network that learns to detect bars in an image.')
    parser.add_argument('--model', type=str, default='hslif', help='the neuron model to evaluate, if, lif, ftlif, exlif, adex, or hslif')
    parser.add_argument('--increment', type=float, default=1.0, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--input_rate', type=float, default=120.0, help='maximum firing rate of input sensory neurons (Hz)')
    parser.add_argument('--stdp', type=str, default='pair', help='form of stdp to use, can be pair or triplet')
    parser.add_argument('--target_frequency', type=float, default=5, help='target frequency in Hz of neuron (only applicable to HSLIF neurons')
    parser.add_argument('--episodes', type=int, default=10000, help='number of episodes')
    parser.add_argument('--nri', type=float, default=3.0, help='noise resampling interval')
    parser.add_argument('--np', type=float, default=0.01, help='noise probability')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--m', type=int, default=150, help='number of past rewards/accuracies to keep track of for averaging')
    parser.add_argument('--viz', action="store_true", help='run graphical version')
    parser.add_argument('--load', action="store_true", help='load saved version')
    parser.add_argument('--load_dir', type=str, default='../datasets', help='path to directory where saved weight matrices are stored')

    # parse user input
    args = parser.parse_args()

    # load the MNIST dataset
    train_data, train_labels, test_data, test_labels = load_mnist()

    # define output neuron classes
    network_labels = [0, 1, 2] 

    ind = np.where(train_labels<len(network_labels))
    train_data = train_data[ind]
    train_labels = train_labels[ind]

    # create the timer
    tki = TimeKeeperIterator(timeunit=args.increment*msec)

    # select neuro  n model based on user input
    if args.model == 'if':
        model = IFNeuralGroup
        e_params = IFParams()
    elif args.model == 'lif':
        model = LIFNeuralGroup
        e_params = LIFParams()
    elif args.model == 'ftlif':
        model = FTLIFNeuralGroup
        e_params = FTLIFParams()
    elif args.model == 'exlif':
        model = ExLIFNeuralGroup
        e_params = ExLIFParams()
    elif args.model == 'adex':
        model = AdExNeuralGroup
        e_params = AdExParams()
    elif args.model == 'hslif':
        model = HSLIFNeuralGroup
        e_params = HSLIFParams()
        # set phi for homeostasis
        e_params.phi = calculate_phi(args.target_frequency, tki)
    else:
        raise RuntimeError("%s is not a valid neuron model, must be if, lif, ftlif, exlif, adex, or hslif." % args.model)
    
    lp = DASTDPParams()         
    lp.lr = args.lr
    lp.ab_et_tao = 0.1 * sec   
    lp.ba_et_tao = 0.1 * sec

    # inhibitory neuron parameters
    i_params = LIFParams()
    i_params.v_thr = -69.0 * mvolt
    i_params.tao_m = 100*msec

    # modify a few excitatory neuron parameters
    e_params.tao_m = 5*msec

    # Define neural groups
    g1 = SensoryNeuralGroup(1, 16*16, "g1", 1, tki, e_params, field_shape=(16, 16), viz_layer_pos=(0,0))

    g2 = model(1, 9, "g2", 2, tki, e_params, field_shape=(3, 3), viz_layer_pos=(0,0))
    g2i = LIFNeuralGroup(0, 9, "g2i", 2, tki, i_params, field_shape=(3, 3), viz_layer_pos=(0,20))

    g3 = model(1, 9, "g3", 2, tki, e_params, field_shape=(3, 3), viz_layer_pos=(0,1))
    g3i = LIFNeuralGroup(0, 9, "g3i", 2, tki, i_params, field_shape=(3, 3), viz_layer_pos=(0,20))

    g4 = model(1, 9, "g4", 2, tki, e_params, field_shape=(3, 3), viz_layer_pos=(0,2))
    g4i = LIFNeuralGroup(0, 9, "g4i", 2, tki, i_params, field_shape=(3, 3), viz_layer_pos=(0,20))

    g5 = model(1, 9, "g5", 2, tki, e_params, field_shape=(3, 3), viz_layer_pos=(0,3))
    g5i = LIFNeuralGroup(0, 9, "g5i", 2, tki, i_params, field_shape=(3, 3), viz_layer_pos=(0,20))

    g6 = model(1, 9, "g6", 2, tki, e_params, field_shape=(3, 3), viz_layer_pos=(0,4))
    g6i = LIFNeuralGroup(0, 9, "g6i", 2, tki, i_params, field_shape=(3, 3), viz_layer_pos=(0,20))

    g7 = model(1, 9, "g7", 2, tki, e_params, field_shape=(3, 3), viz_layer_pos=(0,5))
    g7i = LIFNeuralGroup(0, 9, "g7i", 2, tki, i_params, field_shape=(3, 3), viz_layer_pos=(0,20))

    g8 = model(1, 9, "g8", 2, tki, e_params, field_shape=(3, 3), viz_layer_pos=(0,6))
    g8i = LIFNeuralGroup(0, 9, "g8i", 2, tki, i_params, field_shape=(3, 3), viz_layer_pos=(0,20))

    g9 = model(1, 9, "g9", 2, tki, e_params, field_shape=(3, 3), viz_layer_pos=(0,7))
    g9i = LIFNeuralGroup(0, 9, "g9i", 2, tki, i_params, field_shape=(3, 3), viz_layer_pos=(0,20))

    g10 = model(1, 9, "g10", 2, tki, e_params, field_shape=(3, 3), viz_layer_pos=(0,8))
    g10i = LIFNeuralGroup(0, 9, "g10i", 2, tki, i_params, field_shape=(3, 3), viz_layer_pos=(0,20))

    # gm = model(1, 5*5, "gm", tki, e_params, field_shape=(5, 5))
    # gmi = LIFNeuralGroup(0, 5*5, "gmi", tki, i_params, field_shape=(5, 5))

    gl = LIFNeuralGroup(1, len(network_labels), "gl", 3, tki, e_params)
    gli = LIFNeuralGroup(0, len(network_labels), "gli", 3, tki, i_params)

    # create neural network
    nn = NeuralNetwork([g1, g2, g2i, g3, g3i, g4, g4i, g5, g5i, g6, g6i, g7, g7i, g8, g8i, g9, g9i, g10, g10i, gli, gl], "multilayer_mnist", tki)
    
    # excitatory feed-forward
    nn.local_connect('g1', 'g2', (8,8), 4, 4, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.local_connect('g1', 'g3', (8,8), 4, 4, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.local_connect('g1', 'g4', (8,8), 4, 4, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.local_connect('g1', 'g5', (8,8), 4, 4, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.local_connect('g1', 'g6', (8,8), 4, 4, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.local_connect('g1', 'g7', (8,8), 4, 4, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.local_connect('g1', 'g8', (8,8), 4, 4, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.local_connect('g1', 'g9', (8,8), 4, 4, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.local_connect('g1', 'g10', (8,8), 4, 4, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    # nn.local_connect('g2', 'gm', (4,4), 2, 2, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    # nn.local_connect('g3', 'gm', (4,4), 2, 2, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    # nn.local_connect('g4', 'gm', (4,4), 2, 2, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    # nn.local_connect('g5', 'gm', (4,4), 2, 2, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    # nn.fully_connect('gm', 'gl', trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.fully_connect('g2', 'gl', trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.fully_connect('g3', 'gl', trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.fully_connect('g4', 'gl', trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.fully_connect('g5', 'gl', trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.fully_connect('g6', 'gl', trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.fully_connect('g7', 'gl', trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.fully_connect('g8', 'gl', trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.fully_connect('g9', 'gl', trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.fully_connect('g10', 'gl', trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')

    # inhibitory lateral feedback
    nn.one_to_one_connect("g2", "g2i", w_i=1.0, trainable=False)
    nn.fully_connect("g2i", "g2", w_i=1.0, trainable=False, skip_one_to_one=True)
    nn.fully_connect("g2i", "g3", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g2i", "g4", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g2i", "g5", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g2i", "g6", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g2i", "g7", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g2i", "g8", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g2i", "g9", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g2i", "g10", w_i=1.0, trainable=False, skip_one_to_one=False)

    nn.one_to_one_connect("g3", "g3i", w_i=1.0, trainable=False)
    nn.fully_connect("g3i", "g3", w_i=1.0, trainable=False, skip_one_to_one=True)
    nn.fully_connect("g3i", "g2", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g3i", "g4", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g3i", "g5", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g3i", "g6", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g3i", "g7", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g3i", "g8", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g3i", "g9", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g3i", "g10", w_i=1.0, trainable=False, skip_one_to_one=False)

    nn.one_to_one_connect("g4", "g4i", w_i=1.0, trainable=False)
    nn.fully_connect("g4i", "g4", w_i=1.0, trainable=False, skip_one_to_one=True)
    nn.fully_connect("g4i", "g2", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g4i", "g3", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g4i", "g5", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g4i", "g6", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g4i", "g7", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g4i", "g8", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g4i", "g9", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g4i", "g10", w_i=1.0, trainable=False, skip_one_to_one=False)

    nn.one_to_one_connect("g5", "g5i", w_i=1.0, trainable=False)
    nn.fully_connect("g5i", "g5", w_i=1.0, trainable=False, skip_one_to_one=True)
    nn.fully_connect("g5i", "g2", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g5i", "g3", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g5i", "g4", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g5i", "g6", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g5i", "g7", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g5i", "g8", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g5i", "g9", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g5i", "g10", w_i=1.0, trainable=False, skip_one_to_one=False)

    nn.one_to_one_connect("g6", "g6i", w_i=1.0, trainable=False)
    nn.fully_connect("g6i", "g2", w_i=1.0, trainable=False, skip_one_to_one=True)
    nn.fully_connect("g6i", "g3", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g6i", "g4", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g6i", "g5", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g6i", "g7", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g6i", "g8", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g6i", "g9", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g6i", "g10", w_i=1.0, trainable=False, skip_one_to_one=False)

    nn.one_to_one_connect("g7", "g7i", w_i=1.0, trainable=False)
    nn.fully_connect("g7i", "g2", w_i=1.0, trainable=False, skip_one_to_one=True)
    nn.fully_connect("g7i", "g3", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g7i", "g4", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g7i", "g5", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g7i", "g6", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g7i", "g8", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g7i", "g9", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g7i", "g10", w_i=1.0, trainable=False, skip_one_to_one=False)

    nn.one_to_one_connect("g8", "g8i", w_i=1.0, trainable=False)
    nn.fully_connect("g8i", "g2", w_i=1.0, trainable=False, skip_one_to_one=True)
    nn.fully_connect("g8i", "g3", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g8i", "g4", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g8i", "g5", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g8i", "g6", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g8i", "g7", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g8i", "g9", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g8i", "g10", w_i=1.0, trainable=False, skip_one_to_one=False)

    nn.one_to_one_connect("g9", "g9i", w_i=1.0, trainable=False)
    nn.fully_connect("g9i", "g2", w_i=1.0, trainable=False, skip_one_to_one=True)
    nn.fully_connect("g9i", "g3", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g9i", "g4", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g9i", "g5", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g9i", "g6", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g9i", "g7", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g9i", "g8", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g9i", "g10", w_i=1.0, trainable=False, skip_one_to_one=False)

    nn.one_to_one_connect("g10", "g10i", w_i=1.0, trainable=False)
    nn.fully_connect("g10i", "g2", w_i=1.0, trainable=False, skip_one_to_one=True)
    nn.fully_connect("g10i", "g3", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g10i", "g4", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g10i", "g5", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g10i", "g6", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g10i", "g7", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g10i", "g8", w_i=1.0, trainable=False, skip_one_to_one=False)
    nn.fully_connect("g10i", "g9", w_i=1.0, trainable=False, skip_one_to_one=False)

    nn.one_to_one_connect("gl", "gli", w_i=1.0, trainable=False)
    nn.fully_connect("gli", "gl", w_i=1.0, trainable=False, skip_one_to_one=True)

    if args.load:
        paths = [os.path.join(args.load_dir, fname) for fname in os.listdir(args.load_dir) if fname.endswith('.npy')]
        nn.load_w(paths)
    # deque for tracking various things
    reward_history = deque(args.m*[0], args.m)
    reward_track = []  # list of average rewards
    avg_reward = 0
    accuracy_history = deque(args.m*[0], args.m)
    accuracy_track = []  # list of average accuracy
    avg_accuracy = 0
    num_metrics = 0  # number of times we have appended to the metrics arrays

    nrh = NetworkRunHandler(nn, 
                                train_data,
                                train_labels,
                                network_labels,
                                ("g1", "g2", "g3", "g4", "g5", 'g6', 'g7', 'g8', 'g9', 'g10', "g2i", "g3i", "g4i", "g5i", 'g6i', 'g7i', 'g8i', 'g9i', 'g10i', "gl", "gli"),
                                base_input_frequency=args.input_rate,
                                additive_frequency_boost=5.0,
                                additive_frequency_wait=50.0,
                                enable_noise_sampling=True,
                                noise_sampling_interval=args.nri,
                                noise_probability=args.np,
                                episodes=args.episodes,
                                save_at_end=True,
                                save_dir="C:/Users/james/CODE/Ganglion/ganglion/datasets",
                                reset_on_process=True,
                                rewards=(-1.0, 1.0),
                                data_pre_processor=resize,
                                training=False if args.load else True)
    if args.viz:
        window = VizWindow(nrh, no_show_inhib=True)
        pyglet.app.run()
    else:
        running = True
        while running:
            running, metrics = nrh.run_episode()
            r, a, cs = metrics
            reward_history.append(r)
            accuracy_history.append(a)
            num_metrics += 1
            if num_metrics >= args.m:
                avg_reward = sum(reward_history) / args.m 
                avg_accuracy = sum(accuracy_history) / args.m 
                reward_track.append(avg_reward)
                accuracy_track.append(avg_accuracy)
                print("Episode %s ::  Average Reward %.2f :: Average Accuracy %.2f :: Cummulative Spike Count %s" % (nrh.current_episode-1, avg_reward, avg_accuracy, str(cs)))
            else:
                print("Building reward and accuracy history")

    plt.plot(accuracy_track)
    plt.show()

    # -------------------------------------------------------
    img1 = form_full_weight_map(g2, "g1", "g2", nn)
    img2 = form_full_weight_map(g2, "g1", "g3", nn)
    img3 = form_full_weight_map(g2, "g1", "g4", nn)
    img4 = form_full_weight_map(g2, "g1", "g5", nn)
    img5 = form_full_weight_map(g2, "g1", "g6", nn)
    img6 = form_full_weight_map(g2, "g1", "g7", nn)
    img7 = form_full_weight_map(g2, "g1", "g8", nn)
    img8 = form_full_weight_map(g2, "g1", "g9", nn)
    img9 = form_full_weight_map(g2, "g1", "g10", nn)

    img = np.hstack((img1, img2, img3, img4, img5, img6, img7, img8, img9))

    plt.imshow(img)
    plt.title("Post-training weight matrix")
    plt.colorbar()
    plt.clim(0, img.max())
    plt.show()
    # -------------------------------------------------------

