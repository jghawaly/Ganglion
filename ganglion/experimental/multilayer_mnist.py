import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, IFNeuralGroup, LIFNeuralGroup, FTLIFNeuralGroup, ExLIFNeuralGroup, AdExNeuralGroup, HSLIFNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import IFParams, LIFParams, FTLIFParams, ExLIFParams, AdExParams, HSLIFParams, DASTDPParams
from units import *
from utils import poisson_train, calculate_phi, load_mnist, add_noise

import cv2
import numpy as np 
import numpy.random as nprand
import matplotlib.pyplot as plt
import argparse
import random
from collections import deque


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation of a neural network that learns to detect bars in an image.')
    parser.add_argument('--model', type=str, default='hslif', help='the neuron model to evaluate, if, lif, ftlif, exlif, adex, or hslif')
    parser.add_argument('--increment', type=float, default=1.0, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--exposure', type=float, default=20.0, help='the duration of time that the network is exposed to each training example')
    parser.add_argument('--input_rate', type=float, default=20.0, help='maximum firing rate of input sensory neurons (Hz)')
    parser.add_argument('--stdp', type=str, default='pair', help='form of stdp to use, can be pair or triplet')
    parser.add_argument('--target_frequency', type=float, default=5, help='target frequency in Hz of neuron (only applicable to HSLIF neurons')
    parser.add_argument('--episodes', type=int, default=10000, help='number of episodes')
    parser.add_argument('--chi', type=float, default=0.0, help='a float value betwen 0 and 1 that is the probability of sampling the next action based on relative firing rates')
    parser.add_argument('--nri', type=float, default=3.0, help='noise resampling interval')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--rh', type=int, default=100, help='number of past rewards to keep track of for averaging')
    parser.add_argument('--ah', type=int, default=100, help='number of past accuracy measures to keep track of for averaging')

    # parse user input
    args = parser.parse_args()

    # load the MNIST dataset
    train_data, train_labels, test_data, test_labels = load_mnist()

    ind = np.where(train_labels<3)
    train_data = train_data[ind]
    train_labels = train_labels[ind]

    # define output neuron classes
    # network_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  
    network_labels = [0, 1, 2] 

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

    # inhibitory neuron parameters
    i_params = LIFParams()
    i_params.v_thr = -69.0 * mvolt
    i_params.tao_m = 100*msec

    # modify a few excitatory neuron parameters
    e_params.tao_m = 100*msec

    # Define neural groups
    g1 = SensoryNeuralGroup(1, 16*16, "g1", tki, e_params, field_shape=(16, 16))

    g2 = model(1, 12*12, "g2", tki, e_params, field_shape=(12, 12))
    g2i = LIFNeuralGroup(0, 12*12, "g2i", tki, i_params, field_shape=(12, 12))

    g3 = model(1, 5*5, "g3", tki, e_params, field_shape=(5, 5))
    g3i = LIFNeuralGroup(0, 5*5, "g3i", tki, i_params, field_shape=(5, 5))

    g4 = LIFNeuralGroup(1, len(network_labels), "g4", tki, e_params)
    g4i = LIFNeuralGroup(0, len(network_labels), "g4i", tki, i_params)

    # create neural network
    nn = NeuralNetwork([g1, g2, g2i, g3, g3i, g4, g4i], "multilayer_mnist", tki)
    
    # excitatory feed-forward
    # nn.fully_connect('g1', 'g2', trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.local_connect('g1', 'g2', (5,5), 1, 1, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    # nn.fully_connect("g2", "g3", trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.local_connect('g2', 'g3', (4,4), 2, 2, trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.fully_connect('g3', 'g4', trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')

    # inhibitory lateral feedback
    nn.one_to_one_connect("g2", "g2i", w_i=1.0, trainable=False)
    nn.fully_connect("g2i", "g2", w_i=1.0, trainable=False, skip_one_to_one=True)
    nn.one_to_one_connect("g3", "g3i", w_i=1.0, trainable=False)
    nn.fully_connect("g3i", "g3", w_i=1.0, trainable=False, skip_one_to_one=True)
    nn.one_to_one_connect("g4", "g4i", w_i=1.0, trainable=False)
    nn.fully_connect("g4i", "g4", w_i=1.0, trainable=False, skip_one_to_one=True)

    # pre-normalize weights
    nn.normalize_weights()

    # setup simulation parameters
    last_exposure_step = 1  # timestep of last exposure
    last_resample_step = 1 # timestep of last resampling
    cummulative_spikes = np.zeros(g4.shape)  # initialize the cummulative spikes matrix
    i = 0  # current episode
    ft_add = 0  # additive frequency
    last_result = "       "
    time_since_eval = 0

    # sample initial image and label
    mnist_counter = 0
    img = cv2.resize(add_noise(train_data[mnist_counter]), (16,16), interpolation=cv2.INTER_AREA)
    label = train_labels[mnist_counter]

    # deque for tracking various things
    reward_history = deque(args.rh*[0], args.rh)
    reward_track = []  # list of average rewards
    avg_reward = 0
    accuracy_history = deque(args.ah*[0], args.ah)
    accuracy_track = []  # list of average accuracy
    avg_accuracy = 0

    for step in tki:
        if (step - last_resample_step) * tki.dt() >= args.nri * msec:
            last_resample_step = step
            img = cv2.resize(add_noise(train_data[mnist_counter]), (16,16), interpolation=cv2.INTER_AREA)
        if cummulative_spikes.sum() > 0 and np.count_nonzero(cummulative_spikes == cummulative_spikes.max()) == 1:
            last_exposure_step = step
            dice = random.random()
            action = cummulative_spikes.argmax()

            # True for correct prediction, False otherwise
            correct = network_labels[action] == label
            
            # determine reward
            reward = 1.0 if correct else -1.0

            # get new image and label
            mnist_counter += 1
            if mnist_counter == train_data.shape[0]:
                mnist_counter = 0
            img = cv2.resize(train_data[mnist_counter], (16,16), interpolation=cv2.INTER_AREA)
            label = train_labels[mnist_counter]

            # puff some dopamine into the network
            nn.dopamine_puff(reward, actions=action)
            
            # iterate episode
            i += 1

            # track reward
            reward_history.append(reward)
            avg_reward = sum(reward_history)/args.rh
            reward_track.append(avg_reward)

            # track accuracy
            accuracy_history.append(1.0 if correct else 0.0)
            avg_accuracy = sum(accuracy_history)/args.ah
            accuracy_track.append(avg_accuracy)

            cummulative_spikes.fill(0)

            nn.reset()
            nn.normalize_weights()
            ft_add = 0
        else:
            time_since_eval += tki.dt() 
            if time_since_eval >= 50 * msec:
                ft_add += 5
                time_since_eval = 0

        # inject spikes into sensory layer
        g1.run(poisson_train(img, tki.dt(), args.input_rate + ft_add))

        # run all layers
        nn.run_order(["g1", "g2", "g2i", "g3", "g3i", "g4", "g4i"])

        cummulative_spikes += g4.spike_count
        
        sys.stdout.write("Simulation time :: %.0f ms :: Episode :: %g :: Average Reward :: %.2f :: Average Accuracy :: %.2f  :: # Spikes :: %g                  \r" % (step * tki.dt() / msec, i, avg_reward, avg_accuracy, cummulative_spikes.sum()))

        if i == args.episodes:
            break
    
    print("\n\n")
    # nn.save_w("g1_g2.npy", '1', '2')
    # nn.save_w("g2_g3.npy", '2', '3')
    plt.plot(accuracy_track)
    plt.show()

    # -------------------------------------------------------
    img = form_full_weight_map(g2, "g1", "g2", nn)
    plt.imshow(img)
    plt.title("Post-training weight matrix")
    plt.colorbar()
    plt.clim(0, img.max())
    plt.show()
    # -------------------------------------------------------

    # -------------------------------------------------------
    img = form_full_weight_map(g2, "g2", "g3", nn)
    plt.imshow(img)
    plt.title("Post-training weight matrix")
    plt.colorbar()
    plt.clim(0, img.max())
    plt.show()
    # -------------------------------------------------------

