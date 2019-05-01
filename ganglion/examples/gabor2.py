import sys
sys.path.append("../vectorized")

from utils import load_mnist
from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, FTLIFNeuralGroup, LIFNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import FTLIFParams, STDPParams
from units import *
from utils import poisson_train, load_mnist, save_img

import cv2
import numpy as np
import matplotlib.pyplot as plt

import random
import string
import argparse
import os


def gen_filter_bank(num_orientations, num_wl, kernel_size):
    """
    Generate a filter bank of unique Gabor filters
    """
    angles = np.linspace(0, np.pi-np.pi/num_orientations, num_orientations)
    wl = np.linspace(kernel_size // 4, kernel_size, num_wl)
    filter_bank = []
    for theta in angles:
        wave_bank = []
        for wavelength in wl:
            wave_bank.append(cv2.getGaborKernel((kernel_size, kernel_size), kernel_size/4, theta, wavelength, 0.9, 0))
        filter_bank.append(np.array(wave_bank))
    
    return np.array(filter_bank)

def run_filters(filter_bank, img):
    """
    Apply a bank of filters to the given image and return their total response
    """
    out = np.zeros((filter_bank.shape[0], filter_bank.shape[1]), dtype=np.float)

    # for each orientation
    for i in range(filter_bank.shape[0]):
        # for each wavelength
        for j in range(filter_bank.shape[1]):
            # get the filter
            f = filter_bank[i, j]
            # filter the image
            filtered_img = cv2.filter2D(img, -1, f)
            # plt.imshow(filtered_img)
            # plt.show()
            # calculate the total sum
            response = filtered_img.sum()
                # if i ==0 and j==0:
                #     print(response)
            # set the response
            out[i, j] = response
    return out/out.sum() * 10


def gen_filter_map(filter_bank):
    """
    Generate a displayable form of the filter map
    """
    # for each orientation
    for i in range(filter_bank.shape[0]):
        # for each wavelength
        for j in range(filter_bank.shape[1]):
            if j == 0:
                # get the filter
                f = filter_bank[i, j]
            else:
                f = np.hstack((f, filter_bank[i, j]))
        
        if i == 0:
            tf = f 
        else:
            tf = np.vstack((tf, f))
            
    return tf


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
    parser = argparse.ArgumentParser(description='Simulation of a neural network for classification of MNIST digits.')
    parser.add_argument('--duration', type=float, default=1800000.0, help='duration of simulation (milliseconds)')
    parser.add_argument('--increment', type=float, default=1.0, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--exposure', type=float, default=40.0, help='the duration of time that the network is exposed to each training example')
    parser.add_argument('--input_rate', type=float, default=240.0, help='maximum firing rate of input sensory neurons (Hz)')
    parser.add_argument('--grid_size', type=int, default=10, help='length and width of square grid that bars are generated on')
    parser.add_argument('--stdp', type=str, default='pair', help='form of stdp to use, can be pair or triplet')
    parser.add_argument('--save_increment', type=float, default=-1.0, help='time increment to save an image of the weight map during training (millisecnds)')
    parser.add_argument('--save_dir', type=str, default=".", help='path to directory in which to save weight map images')
    parser.add_argument('--gbar_e', type=float, default=10.0, help='maximum excitatory synapse conductivity (nano-siemens)')
    parser.add_argument('--gbar_i', type=float, default=100.0, help='maximum inhibitory synapse conductivity (nano-siemens)')
    parser.add_argument("--display", action="store_true", help="show final weights at end of simulation")
    parser.add_argument("--save_weights", action="store_true", help="save weights as npy file to the path specified in save_path")
    parser.add_argument("--save_path", type=str, default="./mnist_gabor_weights.npy", help="path to file in which to save the weight matrix")
    parser.add_argument("--test", action="store_true", help="run the network in test mode")
    parser.add_argument("--weights", type=str, default='./saved/mnist_gabor_weights.npy', help="path to numpy weights file to load")
    parser.add_argument("--display_trial", action="store_true", help="show activity of network after each testing trial")
    parser.add_argument("--classes", type=str, default='./saved/mnist_gabor_classes.npy', help='path to file containing class labels for each neuron')
    parser.add_argument("--num_orientations", type=int, default=8, help='number of gabor orientations to use for the filter bank')
    parser.add_argument("--num_wl", type=int, default=5, help='number of gabor wavelengths to use for the filter bank')
    parser.add_argument("--kernel_size", type=int, default=28, help='kernel size of Gabor filter')
    parser.add_argument("--display_filters", action="store_true", help="display the generated filters")

    args = parser.parse_args()

    # Load MNIST data
    train_data, train_labels, test_data, test_labels = load_mnist()

    # Generate Gabor filters
    filters = gen_filter_bank(args.num_orientations, args.num_wl, args.kernel_size)
    
    if args.display_filters:
        m = gen_filter_map(filters)
        plt.imshow(m, cmap='gray')
        plt.title("Gabor Filter Bank")
        plt.show()

    # for d in train_data:
    #     plt.imshow(d)
    #     plt.show()
    #     filtered_mnist = run_filters(filters, d)
    #     plt.imshow(filtered_mnist)
    #     plt.colorbar()
    #     plt.show()
    # define timekeeper
    tki = TimeKeeperIterator(timeunit=args.increment*msec)
    duration = args.duration * msec

    # Define inhibitory neural group parameters
    inhib_layer_params = FTLIFParams()
    inhib_layer_params.gbar_i = args.gbar_i * nsiem
    inhib_layer_params.tao_m = 100 * msec

    # Define excitatory neural group parameters
    exc_layer_params = FTLIFParams()
    exc_layer_params.gbar_e = args.gbar_e * nsiem
    exc_layer_params.tao_m = 100 * msec
    exc_layer_params.tao_ft = 3000 * msec
    exc_layer_params.ft_add = 3 * mvolt

    # create neuron groups
    g1 = SensoryNeuralGroup(np.ones(args.num_orientations * args.num_wl, dtype=np.int), "input", tki, exc_layer_params, field_shape=(args.num_orientations, args.num_wl))
    g2 = FTLIFNeuralGroup(np.ones(args.grid_size * args.grid_size, dtype=np.int), "exc", tki, exc_layer_params, field_shape=(args.grid_size, args.grid_size), forced_wta=True)
    g3 = LIFNeuralGroup(np.zeros(args.grid_size * args.grid_size, dtype=np.int), "inh_lateral", tki, inhib_layer_params, field_shape=(args.grid_size, args.grid_size))

    # create neural network
    nn = NeuralNetwork([g1, g2, g3], "mnist_learner", tki)
    lp = STDPParams()
    lp.lr_pre = 0.001
    lp.lr_post = 0.001

    # create synapses between neural groups
    if not args.test:
        # connect each sensory group to the excitatory layer
        nn.fully_connect("input", "exc", trainable=True, stdp_params=lp, minw=0.5, maxw=0.6, stdp_form=args.stdp)
    else:
        w = np.load(args.weights)
        nn.fully_connect("input", "exc", trainable=False, stdp_params=lp, minw=0.5, maxw=0.6, stdp_form=args.stdp, loaded_weights=w)
    nn.one_to_one_connect("exc", "inh_lateral", w_i=1.0, trainable=False)
    nn.fully_connect("inh_lateral", "exc", w_i=1.0, trainable=False, skip_one_to_one=False)

    if not args.test:
        print("Training begun...")
        # TRAINING MODE
        vms = []

        d = train_data[0]
        last_exposure_step = 0
        last_save_step = 0
        mnist_counter = 0

        sum_spikes = 0
        frequency_addition = 0
        
        # go through each time step
        for step in tki:
            # switch inputs and switch rest state if current exposure duration is greater than the requested time
            if (step - last_exposure_step) * tki.dt() >= args.exposure * msec:
                last_exposure_step = step
                if sum_spikes > 0:
                    # if enough spikes were triggered, then cycle to next training image
                    d = train_data[mnist_counter]
                    mnist_counter += 1

                    # check to see if we went through the entire dataset
                    if mnist_counter >= len(train_data):
                        mnist_counter = 0

                    # force neurons to rest (except for the floating threshold parameter)
                    nn.reset()

                    sum_spikes = 0
                    frequency_addition = 0
                else:
                    # if no spikes were triggered, then increase the input firing rate
                    sum_spikes = 0
                    frequency_addition += 10
            
            # if save requested
            if args.save_increment > 0.0:
                # save image at given increment
                if (step - last_save_step) * tki.dt() >= args.save_increment * msec:
                    last_save_step = step 
                    save_img("%s/%s.bmp" % (args.save_dir, str(mnist_counter)), form_full_weight_map(g2, "input", "exc", nn), normalize=True)
            
            # get Gabor filtered version of image
            filtered_mnist = run_filters(filters, d)

            g1.run(poisson_train(filtered_mnist, tki.dt(), args.input_rate + frequency_addition))
            
            # run all layers
            nn.run_order(['input', 'exc', 'inh_lateral'])

            # normalize weights between sensory groups and g2 so that we don't get weight growth runaway
            for s in nn.synapses:
                if s.trainable:
                    s.scale_weights()
                
            # count the spikes
            sum_spikes += g2.spike_count.sum()

            # report progress
            sys.stdout.write("Current simulation time :: %g milliseconds :: Total Spikes :: %g          \r" % (step * tki.dt() / msec, sum_spikes))
            
            # if simulation time limit is over, then save the weights (if requested) and exit
            if step >= duration/tki.dt():
                if args.save_weights:
                    for s in nn.synapses:
                        if s.post_n == g2 and s.pre_n == g1:
                            s.save_weights(args.save_path)
                break
        
        print("\n\n")

        # if requested, plot and show the final weights between g1 and g2
        if args.display:
            img = form_full_weight_map(g2, "input", "exc", nn)
            plt.imshow(img)
            plt.title("Post-training weight matrix")
            plt.colorbar()
            plt.show()
    else:
        if os.path.exists(args.classes):
            # load provided neuron labels if they exist
            neuron_labels = np.load(args.classes)
        else:
            print("Generating class labels, testing for 1000 mnist examples, this may take a while...")

            # this will keep track of the number of spikes triggered for each neuron, for each class label
            spike_log = np.zeros((args.grid_size * args.grid_size, 10), dtype=np.int)

            d = train_data[0]
            l = train_labels[0]
            last_exposure_step = 0
            mnist_counter = 0

            cummulative_activity = g2.spike_count.copy()
            
            # go through each time step
            for step in tki:
                # switch inputs and switch rest state if current exposure duration is greater than the requested time
                if (step - last_exposure_step) * tki.dt() >= args.exposure * msec:
                    # update spike log
                    for idx, val in np.ndenumerate(cummulative_activity):
                        spike_log[idx, l] += val

                    last_exposure_step = step
                    d = train_data[mnist_counter]
                    l = train_labels[mnist_counter]
                    mnist_counter += 1

                    # force neurons to rest (except for the floating threshold parameter)
                    nn.reset()

                    cummulative_activity.fill(0)
                
                # get Gabor filtered version of image
                filtered_mnist = run_filters(filters, d)

                g1.run(poisson_train(filtered_mnist, tki.dt(), args.input_rate))

                cummulative_activity += g2.spike_count.copy()
                
                # run all layers
                nn.run_order(["input", "exc", "inh_lateral"])

                sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))
                
                # if simulation time limit is over, then exit
                if mnist_counter == 999:
                    break
            
            print("\n\n")
            
            neuron_labels = np.argmax(spike_log, axis=1)
            np.save(args.classes, neuron_labels)
            
            print("Done generating class labels, saved to %s" % args.classes)
        
        print("Testing begun...")
        # TESTING MODE
        d = test_data[0]
        l = test_labels[0]
        last_exposure_step = 0
        mnist_counter = 0

        # tracks neuronal activity for each trial
        cummulative_activity = g2.spike_count.copy()

        # tracks the number of correct predictions
        num_correct = 0
        
        # go through each time step
        for step in tki:
            # switch inputs and switch rest state if current exposure duration is greater than the requested time
            if (step - last_exposure_step) * tki.dt() >= args.exposure * msec:
                # some of the inputs are just too weak to invoke any spikes, so if they don't, just skip it
                if cummulative_activity.sum() > 0:
                    # get labels of active neurons
                    active_labels = neuron_labels[np.where(cummulative_activity > 0)]

                    if active_labels.shape[0] == 2:
                        # if there are only two active neurons, pick the higher one (this is just arbitrary)
                        prediction = np.max(active_labels)
                    else:
                        # get the most frequent label and set it as the prediction
                        prediction = np.argmax(np.bincount(active_labels))

                    if l == prediction:
                        num_correct += 1

                    if args.display_trial:
                        plt.imshow(cummulative_activity.reshape((args.grid_size, args.grid_size)))
                        plt.title("True Label :: %g :: Predicted Label :: %g" % (l, prediction))
                        plt.colorbar()
                        plt.show()
                        

                last_exposure_step = step
                d = test_data[mnist_counter]
                l = test_labels[mnist_counter]
                mnist_counter += 1

                # force neurons to rest (except for the floating threshold parameter)
                nn.reset()

                cummulative_activity.fill(0)
            
            # get Gabor filtered version of image
            filtered_mnist = run_filters(filters, d)

            g1.run(poisson_train(filtered_mnist, tki.dt(), args.input_rate))

            cummulative_activity += g2.spike_count.copy()
            
            # run all layers
            nn.run_order(["input", "exc", "inh_lateral"])

            if mnist_counter > 0:
                sys.stdout.write("Current accuracy :: %.1f %%  :: Number of Evaluations :: %g                    \r" % (100.0 * num_correct / mnist_counter, mnist_counter))

            # if simulation time limit is over, then exit
            if step >= duration/tki.dt():
                break
        
        print("\n\n")
