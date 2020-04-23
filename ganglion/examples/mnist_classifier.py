import sys
sys.path.append("../base")

from timekeeper import TimeKeeperIterator
from NeuralGroup import *
from NeuralNetwork import NeuralNetwork
from NetworkRunHandler import NetworkRunHandler
from parameters import *
from units import *
from utils import *
from viz import VizWindow
import pyglet

import cv2
import numpy as np 
import numpy.random as nprand
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import argparse
import random
from collections import deque
import os
from skimage.morphology import binary_erosion


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

    return img / img.max()


def pre_process(img):
    img = img[4:24, 4:24]

    img = dog(img, 1, 2).clip(min=1e-5)

    # img = apply_log_filter(img, f)
    # img = img.clip(min=1E-5)
    img = unit_normalize(img)
    return img

# def pre_process(img):
#     img = img[4:24, 4:24]
#     img = unit_normalize(img)
#     # img[img<0.8] = 0
#     return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation of a neural network that learns to detect bars in an image.')
    parser.add_argument('--increment', type=float, default=0.15, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--input_rate', type=float, default=1000.0, help='maximum firing rate of input sensory neurons (Hz)')
    parser.add_argument('--episodes', type=int, default=1000, help='number of episodes')
    parser.add_argument('--nri', type=float, default=3.0, help='noise resampling interval')
    parser.add_argument('--np', type=float, default=0.0001, help='noise probability')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.98, help='weight decay value')
    parser.add_argument('--m', type=int, default=30, help='number of past rewards/accuracies to keep track of for averaging')
    parser.add_argument('--viz', action="store_true", help='run graphical version')
    parser.add_argument('--load', action="store_true", help='load saved version')
    parser.add_argument('--sup_val', type=float, default=10, help='supervisory threshold drop')
    parser.add_argument('--load_dir', type=str, default='../models/mnist', help='path to directory where saved weight matrices are stored')
    parser.add_argument('--notrain', action='store_true', help='disables training')
    parser.add_argument('--supervised', action='store_true', help='enables ftm supervision')
    parser.add_argument('--save', action='store_true', help='saves model at end')
    parser.add_argument('--save_dir', type=str, default='../models/mnist/clas_exp', help='path to directory where to save weights')
    parser.add_argument('--total_averaging', action='store_true', help='calculate average accuracy over all data')
    parser.add_argument('--test', action='store_true', help='evaluate on test set')
    parser.add_argument('--justplotfeatures', action='store_true', help="just plot the feature maps and do nothing else")
    parser.add_argument('--gen_cmat', action='store_true', help='generate confusion matrix')
    parser.add_argument('--ext', type=str, default="", help='')

    # parse user input
    args = parser.parse_args()

    # load the MNIST dataset
    
    smnist = StructuredMNIST((0,1,2,3,4,5,6,7,8,9), dataset_dir='../datasets/deskewed_mnist', test=args.test)
    data, labels = smnist.all()

    # define output neuron classes
    network_labels = [0,1,2,3,4,5,6,7,8,9] 

    # create the timer
    tki = TimeKeeperIterator(timeunit=args.increment*msec)

    # main neuron model
    model = AMLIFNeuralGroup
    hidden_params = AMLIFParams()
    
    dap = DASTDPParams()  
    dap.lr = args.lr
    dap.wd=args.wd
    # time constants of eligbility traces
    dap.ab_et_tao = 10 * msec   
    dap.ba_et_tao = 10 * msec
    # scaling factors for positive rewards
    dap.ab_scale_pos = 1.0
    dap.ba_scale_pos = 1.0
    # scaling factors for negative rewards
    dap.ab_scale_neg = -1.0
    dap.ba_scale_neg = -1.0

    # modify input layer neuron parameters
    input_params = IFParams()
    input_params.gbar_e = 1.5 * nsiem   # norm1.5
    input_params.gbar_i = 100 * nsiem
    input_params.vrev_i = -70.0 * mvolt

    # modify hidden layer neuron parameters
    hidden_params.tao_m = 20 * msec  # was 15
    hidden_params.gbar_e = 0.1 * nsiem
    hidden_params.vrev_i = -70.0 * mvolt

    # modify output layer neuron parameters
    output_params = AMLIFParams()
    output_model = model
    if args.supervised:
        output_params = STLIFParams()
        output_model = STLIFNeuralGroup
        output_params.dftm = -1 * args.sup_val * mvolt
        output_params.tao_ftm = 100.0 * msec
    output_params.tao_m = 100 * msec

    # synapse params
    s_params = SynapseParams()
    s_params.tao_syn_e = 0.5 * msec
    s_params.tao_syn_i = 5 * msec
    s_params.spike_window = 10 * msec  # NEW

    # lateral inhibition synapse params
    s_params_lati = SynapseParams()
    s_params_lati.lati = True

    # set our convolutional parameters
    ks = (6,6)  # kernel sizes
    fs = (8,8)  # filter sizes
    stride = 2

    # Define neural groups
    g1 = SensoryNeuralGroup(1, 20*20, "g1", 1, tki, input_params, field_shape=(20, 20), viz_layer_pos=(0,0))
    g2 = model(1,  fs[0]*fs[1], "g2", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,0))
    g3 = model(1,  fs[0]*fs[1], "g3", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,1))
    g4 = model(1,  fs[0]*fs[1], "g4", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,2))
    g5 = model(1,  fs[0]*fs[1], "g5", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,3))
    g6 = model(1,  fs[0]*fs[1], "g6", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,4))
    g7 = model(1,  fs[0]*fs[1], "g7", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,5))
    g8 = model(1,  fs[0]*fs[1], "g8", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,6))
    g9 = model(1,  fs[0]*fs[1], "g9", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,7))
    g10 = model(1, fs[0]*fs[1], "g10", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,8))
    g11 = model(1, fs[0]*fs[1], "g11", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(0,9))
    g12 = model(1, fs[0]*fs[1], "g12", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,0))
    g13 = model(1, fs[0]*fs[1], "g13", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,1))
    g14 = model(1, fs[0]*fs[1], "g14", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,2))
    g15 = model(1, fs[0]*fs[1], "g15", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,3))
    g16 = model(1, fs[0]*fs[1], "g16", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,4))
    g17 = model(1, fs[0]*fs[1], "g17", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,5))
    g18 = model(1, fs[0]*fs[1], "g18", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,6))
    g19 = model(1, fs[0]*fs[1], "g19", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,7))
    g20 = model(1, fs[0]*fs[1], "g20", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,8))
    g21 = model(1, fs[0]*fs[1], "g21", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(1,9))
    g22 = model(1, fs[0]*fs[1], "g22", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(2,9))  #
    g23 = model(1, fs[0]*fs[1], "g23", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(2,9))  #
    g24 = model(1, fs[0]*fs[1], "g24", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(2,9))  #
    g25 = model(1, fs[0]*fs[1], "g25", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(2,9))  #
    g26 = model(1, fs[0]*fs[1], "g26", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(2,9))  #
    g27 = model(1, fs[0]*fs[1], "g27", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(2,9))  #
    g28 = model(1, fs[0]*fs[1], "g28", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(2,9))  #
    g29 = model(1, fs[0]*fs[1], "g29", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(2,9))  #
    g30 = model(1, fs[0]*fs[1], "g30", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(2,9))  #
    g31 = model(1, fs[0]*fs[1], "g31", 2, tki, hidden_params, field_shape=fs, viz_layer_pos=(2,9))  #
    gl = output_model(1, len(network_labels), "gl", 3, tki, output_params)

    # create neural network
    nn = NeuralNetwork([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11, g12, g13, g14, g15, g16, g17, g18, g19, g20, g21, g22, g23, g24, g25, g26, g27, g28, g29, g30, g31, gl], "multilayer_mnist", tki)
    
    # excitatory feed-forward
    nn.multi_local_connect(['g1'], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"],
                            ks, stride, stride, trainable=False, stdp_params=PairSTDPParams(), s_type='base', share_weights=True, syn_params=s_params)
    nn.multi_fully_connect(["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], ['gl'], 
                            trainable=True, stdp_params=dap, minw=0.1, maxw=0.3, s_type='da', syn_params=s_params)

    # inhibitory weight value
    w_inhib = 1.0

    nn.fully_connect("gl", "gl", w_i=w_inhib, s_type='inhib', skip_one_to_one=True)

    # lateral-inhibition
    lati_pars = {'s_type': 'lati'}
    nn.multi_one_to_one_connect(["g2"],  ["g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g3"],  ["g2", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g4"],  ["g2", "g3", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g5"],  ["g2", "g3", "g4", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g6"],  ["g2", "g3", "g4", "g5", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g7"],  ["g2", "g3", "g4", "g5", "g6", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g8"],  ["g2", "g3", "g4", "g5", "g6", "g7", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g9"],  ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g10"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g11"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g12"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g13"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g14"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g15"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g16"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g17"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g18"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g19"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g20"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g21"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g22"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g23"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g24"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g25", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g25"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g26", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g26"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g27", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g27"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g28", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g28"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g29", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g29"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g30", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g30"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g31"], **lati_pars)
    nn.multi_one_to_one_connect(["g31"], ["g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9",  "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30"], **lati_pars)

    if args.load:
        nn.load_w(args.load_dir)
    
    if args.justplotfeatures:
        fig = plt.figure(figsize=(8., 4.))
        grid = ImageGrid(fig, 111,  
                 nrows_ncols=(5, 6),  
                 axes_pad=0.1,  
                 )

        grid[0].imshow(form_full_weight_map(g2, "g1", "g2", nn), cmap='inferno')
        grid[1].imshow(form_full_weight_map(g3, "g1", "g3", nn), cmap='inferno')
        grid[2].imshow(form_full_weight_map(g4, "g1", "g4", nn), cmap='inferno')
        grid[3].imshow(form_full_weight_map(g5, "g1", "g5", nn), cmap='inferno')
        grid[4].imshow(form_full_weight_map(g6, "g1", "g6", nn), cmap='inferno')
        grid[5].imshow(form_full_weight_map(g7, "g1", "g7", nn), cmap='inferno')
        grid[6].imshow(form_full_weight_map(g8, "g1", "g8", nn), cmap='inferno')
        grid[7].imshow(form_full_weight_map(g9, "g1", "g9", nn), cmap='inferno')
        grid[8].imshow(form_full_weight_map(g10, "g1", "g10", nn), cmap='inferno')
        grid[9].imshow(form_full_weight_map(g11, "g1", "g11", nn), cmap='inferno')

        grid[10].imshow(form_full_weight_map(g12, "g1", "g12", nn), cmap='inferno')
        grid[11].imshow(form_full_weight_map(g13, "g1", "g13", nn), cmap='inferno')
        grid[12].imshow(form_full_weight_map(g14, "g1", "g14", nn), cmap='inferno')
        grid[13].imshow(form_full_weight_map(g15, "g1", "g15", nn), cmap='inferno')
        grid[14].imshow(form_full_weight_map(g16, "g1", "g16", nn), cmap='inferno')
        grid[15].imshow(form_full_weight_map(g17, "g1", "g17", nn), cmap='inferno')
        grid[16].imshow(form_full_weight_map(g18, "g1", "g18", nn), cmap='inferno')
        grid[17].imshow(form_full_weight_map(g19, "g1", "g19", nn), cmap='inferno')
        grid[18].imshow(form_full_weight_map(g20, "g1", "g20", nn), cmap='inferno')
        grid[19].imshow(form_full_weight_map(g21, "g1", "g21", nn), cmap='inferno')

        grid[20].imshow(form_full_weight_map(g22, "g1", "g21", nn), cmap='inferno')
        grid[21].imshow(form_full_weight_map(g23, "g1", "g22", nn), cmap='inferno')
        grid[22].imshow(form_full_weight_map(g24, "g1", "g23", nn), cmap='inferno')
        grid[23].imshow(form_full_weight_map(g25, "g1", "g24", nn), cmap='inferno')
        grid[24].imshow(form_full_weight_map(g26, "g1", "g25", nn), cmap='inferno')
        grid[25].imshow(form_full_weight_map(g27, "g1", "g26", nn), cmap='inferno')
        grid[26].imshow(form_full_weight_map(g28, "g1", "g27", nn), cmap='inferno')
        grid[27].imshow(form_full_weight_map(g29, "g1", "g28", nn), cmap='inferno')
        grid[28].imshow(form_full_weight_map(g30, "g1", "g29", nn), cmap='inferno')
        grid[29].imshow(form_full_weight_map(g31, "g1", "g30", nn), cmap='inferno')

        # plt.colorbar()
        # plt.clim(0, img.max())
        plt.show()
        exit()

    # deque for tracking various things
    reward_history = deque(args.m*[0], args.m)
    reward_track = []  # list of average rewards
    avg_reward = 0
    accuracy_history = deque(args.m*[0], args.m)
    accuracy_track = []  # list of average accuracy
    avg_accuracy = 0
    num_metrics = 0  # number of times we have appended to the metrics arrays

    nrh = NetworkRunHandler(nn, 
                            data,
                            labels,
                            network_labels,
                            ("g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9", "g10", "g11", "g12", "g13", "g14", "g15", "g16", "g17", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25", "g26", "g27", "g28", "g29", "g30", "g31", "gl"),
                            base_input_frequency=args.input_rate,
                            enable_additive_frequency=True,
                            additive_frequency_boost=500.0,
                            additive_frequency_wait=10.0,
                            enable_noise_sampling=True,
                            noise_sampling_interval=args.nri,
                            noise_probability=args.np,
                            normalize_on_weight_change=True,
                            normalize_on_start=False if args.notrain else True,
                            episodes=args.episodes,
                            save_at_end=False,
                            reset_on_process=True,
                            rewards=(-1.0, 1.0),
                            data_pre_processor=pre_process,
                            training=False if args.notrain else True,
                            output_encoding=4,
                            encoding_count=1,
                            move_on_time=300*msec,
                            ros=False,
                            output_is_ftm=True if args.supervised else False,
                            ftm_supervised=True if args.supervised else False,
                            verbose=True,
                            store_predictions=True,
                            decay_weights=True)

    if args.viz:
        window = VizWindow(nrh, no_show_inhib=True)
        pyglet.app.run()
    else:
        try:
            total = []
            reward = 0
            accuracy = 0
            cum_reward = []
            cum_accuracy = []
            eps = []
            running = True
            while running:
                # show_full_weight_map(g3, 'exc', 'outputs', nn)
                running, metrics = nrh.run_episode()
                try:
                    r, a, cs = metrics
                except TypeError:
                    r = -1
                    a = 0
                    cs = 0
                
                reward -= reward / args.m
                reward += r / args.m

                accuracy -= accuracy / args.m
                accuracy += a / args.m

                total.append(a)

                num_metrics += 1
                if num_metrics >= args.m:
                    eps.append(nrh.current_episode)
                    cum_reward.append(reward)
                    cum_accuracy.append(accuracy)
                    if args.total_averaging:
                        print("Episode %s ::  Average Accuracy %.2f :: Cummulative Spike Count %s" % (nrh.current_episode-1, np.sum(np.array(total)) / len(total), str(cs)))
                    else:
                        print("Episode %s ::  Average Reward %.2f :: Average Accuracy %.2f :: Cummulative Spike Count %s" % (nrh.current_episode-1, reward, accuracy, str(cs)))
                else:
                    print("Building reward and accuracy history")
            np.save('eps.npy', eps)
            np.save('acc.npy', cum_accuracy)
            plt.plot(eps, cum_accuracy, color='k', linestyle='-', marker='.')
            plt.xlabel("Episode")
            plt.ylabel("Average Accuracy (N=30)")
            plt.show()
        except KeyboardInterrupt:
            exit()
    
    if args.save:
        nn.save_w(args.save_dir)
    
    if args.test:
        y_true, y_pred = nrh.get_predictions()
        np.save(args.ext + 'y_true.npy', y_true)
        np.save(args.ext + 'y_pred.npy', y_pred)
