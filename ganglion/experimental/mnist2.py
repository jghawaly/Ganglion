import sys
sys.path.append("../base")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, IFNeuralGroup, LIFNeuralGroup, FTLIFNeuralGroup, ExLIFNeuralGroup, AdExNeuralGroup, HSLIFNeuralGroup, FTMLIFNeuralGroup
from NeuralNetwork import NeuralNetwork
from NetworkRunHandler import NetworkRunHandler
from parameters import IFParams, LIFParams, FTLIFParams, ExLIFParams, AdExParams, HSLIFParams, DASTDPParams, TripletSTDPParams, PairSTDPParams, FTMLIFParams
from units import *
from utils import *
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


f = log_filter(0.5)
def pre_process(img):
    # img[img<0.9] = 0.0
    img = apply_log_filter(img, f)
    img = unit_normalize(img)
    img = cv2.resize(img, (20,20), interpolation=cv2.INTER_AREA)
    # img = dog_filter(img)
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation of a neural network that learns to detect bars in an image.')
    parser.add_argument('--model', type=str, default='hslif', help='the neuron model to evaluate, if, lif, ftlif, exlif, adex, or hslif')
    parser.add_argument('--increment', type=float, default=1.0, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--input_rate', type=float, default=1000.0, help='maximum firing rate of input sensory neurons (Hz)')
    parser.add_argument('--stdp', type=str, default='pair', help='form of stdp to use, can be pair or triplet')
    parser.add_argument('--target_frequency', type=float, default=20, help='target frequency in Hz of neuron (only applicable to HSLIF neurons')
    parser.add_argument('--episodes', type=int, default=10000, help='number of episodes')
    parser.add_argument('--nri', type=float, default=3.0, help='noise resampling interval')
    parser.add_argument('--np', type=float, default=0.0001, help='noise probability')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--m', type=int, default=100, help='number of past rewards/accuracies to keep track of for averaging')
    parser.add_argument('--dftm', type=float, default=0.001, help='FTM neuron modulation amount')
    parser.add_argument('--viz', action="store_true", help='run graphical version')
    parser.add_argument('--load', action="store_true", help='load saved version')
    parser.add_argument('--load_dir', type=str, default='../datasets/poop', help='path to directory where saved weight matrices are stored')
    parser.add_argument('--notrain', action='store_true', help='disables training')
    parser.add_argument('--nosave', action='store_true', help='disables model saving')
    parser.add_argument('--noplot', action='store_true', help='disables plotting at end of script')
    parser.add_argument('--supervised', action='store_true', help='enables ftm supervision')

    # parse user input
    args = parser.parse_args()

    # load the MNIST dataset
    train_data, train_labels, test_data, test_labels = load_mnist()

    # define output neuron classes
    network_labels = [0,1,2,3,4,5,6,7,8,9] 

    # ind = np.where(train_labels<len(network_labels))
    # train_data = train_data[ind]
    # train_labels = train_labels[ind]

    smnist = StructuredMNIST((0,1,2,3,4,5,6,7,8,9))
    train_data, train_labels = smnist.all()

    # create the timer
    tki = TimeKeeperIterator(timeunit=args.increment*msec)

    # select neuro  n model based on user input
    if args.model == 'if':
        model = IFNeuralGroup
        hidden_params = IFParams()
    elif args.model == 'lif':
        model = LIFNeuralGroup
        hidden_params = LIFParams()
    elif args.model == 'ftlif':
        model = FTLIFNeuralGroup
        hidden_params = FTLIFParams()
    elif args.model == 'exlif':
        model = ExLIFNeuralGroup
        hidden_params = ExLIFParams()
    elif args.model == 'adex':
        model = AdExNeuralGroup
        hidden_params = AdExParams()
    elif args.model == 'hslif':
        model = HSLIFNeuralGroup
        hidden_params = HSLIFParams()
        # set phi for homeostasis
        hidden_params.phi = calculate_phi(args.target_frequency, tki)
    else:
        raise RuntimeError("%s is not a valid neuron model, must be if, lif, ftlif, exlif, adex, or hslif." % args.model)
    
    lp = DASTDPParams()         
    lp.lr = args.lr
    lp.ab_et_tao = 25 * msec  
    lp.ba_et_tao = 25 * msec 

    # model = LIFNeuralGroup
    # modify a few excitatory neuron parameters
    hidden_params.tao_m = 10*msec
    hidden_params.gbar_e = 10 * nsiem  # 20 for smaller input
    hidden_params.gbar_i = 10 * nsiem

    output_params = FTMLIFParams()
    output_params.tao_m = 100 * msec 
    output_params.tao_ftm = 100.0 * msec
    output_params.dftm = args.dftm
    output_params.v_thr = -20.4 * mvolt

    # set our convolutional parameters
    ks = (5,5)  # kernel sizes
    fs = (6,6)  # filter sizes
    stride = 3

    # Define neural groups
    g1 = SensoryNeuralGroup(1, 20*20, "g1", 1, tki, hidden_params, field_shape=(20, 20), viz_layer_pos=(0,0))
    g2 = model(1, fs[0]*fs[1], "g2", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,0))
    g3 = model(1, fs[0]*fs[1], "g3", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,1))
    g4 = model(1, fs[0]*fs[1], "g4", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,2))
    g5 = model(1, fs[0]*fs[1], "g5", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,3))
    g6 = model(1, fs[0]*fs[1], "g6", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,4))
    g7 = model(1, fs[0]*fs[1], "g7", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,5))
    g8 = model(1, fs[0]*fs[1], "g8", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,6))
    g9 = model(1, fs[0]*fs[1], "g9", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,7))
    g10 = model(1, fs[0]*fs[1], "g10", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,8))
    g11 = model(1, fs[0]*fs[1], "g11", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,0))
    g12 = model(1, fs[0]*fs[1], "g12", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,1))
    g13 = model(1, fs[0]*fs[1], "g13", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,2))
    g14 = model(1, fs[0]*fs[1], "g14", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,3))
    g15 = model(1, fs[0]*fs[1], "g15", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,4))
    g16 = model(1, fs[0]*fs[1], "g16", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,5))
    g17 = model(1, fs[0]*fs[1], "g17", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,6))
    g18 = model(1, fs[0]*fs[1], "g18", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,7))
    g19 = model(1, fs[0]*fs[1], "g19", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,8))
    g20 = model(1, fs[0]*fs[1], "g20", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,9))
    if not args.notrain:
        gl = FTMLIFNeuralGroup(1, len(network_labels), "gl", 3, tki, output_params)
    else:
        gl = LIFNeuralGroup(1, len(network_labels), "gl", 3, tki, output_params)

    # create neural network
    nn = NeuralNetwork([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18, g19, g20, gl], "multilayer_mnist", tki)
    
    # excitatory feed-forward
    nn.multi_local_connect(['g1'], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20'], ks, stride, stride, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.multi_fully_connect(["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20'], ['gl'], trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')

    # self-inhibition
    nn.fully_connect("g2", "g2", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g3", "g3", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g4", "g4", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g5", "g5", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g6", "g6", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g7", "g7", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g8", "g8", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g9", "g9", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g10", "g10", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g11", "g11", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g12", "g12", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g13", "g13", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g14", "g14", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g15", "g15", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g16", "g16", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g17", "g17", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g18", "g18", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g19", "g19", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("g20", "g20", w_i=1.0, s_type='inhib', skip_one_to_one=True)
    nn.fully_connect("gl", "gl", w_i=1.0, s_type='inhib', skip_one_to_one=True)

    # lateral-inhibition
    nn.multi_one_to_one_connect(["g2"], ["g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g3"], ["g2", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g4"], ["g2", "g3", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g5"], ["g2", "g3", "g4", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g6"], ["g2", "g3", "g4", "g5", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g7"], ["g2", "g3", "g4", "g5", "g6", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g8"], ["g2", "g3", "g4", "g5", "g6", "g7", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g9"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g10"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g11", "g12", "g13", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g11"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g12", "g13", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g12"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g13", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g13"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g14"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g15", "g16", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g15"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g16", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g16"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g17", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g17"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", 'g18', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g18"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g17", 'g17', 'g19', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g19"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g17", 'g17', 'g18', 'g20'], w_i=1.0, s_type='inhib')
    nn.multi_one_to_one_connect(["g20"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g17", 'g17', 'g18', 'g19'], w_i=1.0, s_type='inhib')

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
                                ("g1", "g2", "g3", "g4", "g5", 'g6', 'g7', 'g8', 'g9', 'g10', "g11", "g12", "g13", "g14", "g15", "g16", "g17", 'g18', 'g19', 'g20', "gl"),
                                base_input_frequency=args.input_rate,
                                enable_additive_frequency=True,
                                additive_frequency_boost=100.0,
                                additive_frequency_wait=20.0,
                                enable_noise_sampling=True,
                                noise_sampling_interval=args.nri,
                                noise_probability=args.np,
                                episodes=args.episodes,
                                save_at_end=True if not args.nosave else False,
                                save_dir="../datasets/poop",
                                reset_on_process=True,
                                rewards=(-1.0, 1.0),
                                data_pre_processor=pre_process,
                                training=False if args.notrain else True,
                                output_encoding=1,
                                output_is_ftm=True if not args.notrain else False,
                                ftm_supervised=args.supervised,
                                normalize_on_weight_change=True,
                                normalize_on_start=True)
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

    if not args.noplot:
        plt.plot(accuracy_track)
        plt.show()
        img1 = form_full_weight_map(g2, "g1", "g2", nn)
        img2 = form_full_weight_map(g2, "g1", "g3", nn)
        img3 = form_full_weight_map(g2, "g1", "g4", nn)
        img4 = form_full_weight_map(g2, "g1", "g5", nn)
        img5 = form_full_weight_map(g2, "g1", "g6", nn)
        img6 = form_full_weight_map(g2, "g1", "g7", nn)
        img7 = form_full_weight_map(g2, "g1", "g8", nn)
        img8 = form_full_weight_map(g2, "g1", "g9", nn)
        img9 = form_full_weight_map(g2, "g1", "g10", nn)

        img10 = form_full_weight_map(g2, "g1", "g11", nn)
        img11 = form_full_weight_map(g2, "g1", "g12", nn)
        img12 = form_full_weight_map(g2, "g1", "g13", nn)
        img13 = form_full_weight_map(g2, "g1", "g14", nn)
        img14 = form_full_weight_map(g2, "g1", "g15", nn)
        img15 = form_full_weight_map(g2, "g1", "g16", nn)
        img16 = form_full_weight_map(g2, "g1", "g17", nn)
        img17 = form_full_weight_map(g2, "g1", "g18", nn)
        img18 = form_full_weight_map(g2, "g1", "g19", nn)

        img1 = np.hstack((img1, img2, img3, img4, img5, img6, img7, img8, img9))
        img2 = np.hstack((img10, img11, img12, img13, img14, img15, img16, img17, img18))
        img = np.vstack((img1, img2))

        plt.imshow(img)
        plt.title("Post-training weight matrix")
        plt.colorbar()
        plt.clim(0, img.max())
        plt.show()

