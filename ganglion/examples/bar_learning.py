import sys
sys.path.append("../base")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, IFNeuralGroup, LIFNeuralGroup, FTLIFNeuralGroup, ExLIFNeuralGroup, AdExNeuralGroup, HSLIFNeuralGroup
from NeuralNetwork import NeuralNetwork
from NetworkRunHandler import NetworkRunHandler
from parameters import IFParams, LIFParams, FTLIFParams, ExLIFParams, AdExParams, PairSTDPParams, TripletSTDPParams, HSLIFParams
from units import *
from utils import poisson_train, calculate_phi
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import argparse
from viz import VizWindow
import pyglet


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

def gen_bar_dataset(s):
    data = []
    labels = []
    for i in range(2):
        for j in range(s):
            img = np.zeros((s, s), dtype=np.float)
            if i == 0:  # horizontal
                img[:, j] = 1.0
            else:  # vertical
                img[j, :] = 1.0
            
            data.append(img / img.sum())
            labels.append("%s_%s" % (str(s), str(i)))
    
    np.save('../datasets/bars/bar_data.npy', data)
    np.save('../datasets/bars/bar_labels.npy', labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation of a neural network that learns to detect bars in an image.')
    parser.add_argument('--model', type=str, default='hslif', help='the neuron model to evaluate, if, lif, ftlif, exlif, adex, or hslif')
    # parser.add_argument('--duration', type=float, default=100000.0, help='duration of simulation (milliseconds)')
    parser.add_argument('--increment', type=float, default=0.5, help='time increment for numerical integration (milliseconds)')
    # parser.add_argument('--exposure', type=float, default=40.0, help='the duration of time that the network is exposed to each training example')
    # parser.add_argument('--input_rate', type=float, default=64.0, help='maximum firing rate of input sensory neurons (Hz)')
    parser.add_argument('--grid_size', type=int, default=16, help='length and width of square grid that bars are generated on')
    parser.add_argument('--stdp', type=str, default='pair', help='form of stdp to use, can be pair or triplet')
    parser.add_argument('--target_frequency', type=float, default=2, help='target frequency in Hz of neuron (only applicable to HSLIF neurons.')
    parser.add_argument('--e', type=int, default=640, help='number of episodes to run')

    args = parser.parse_args()
    # Define the time keeper and total duration of training
    tki = TimeKeeperIterator(timeunit=args.increment*msec)
    # duration = args.duration * msec

    # select the requested neuron model and neuron model parameters
    if args.model == 'if':
        model = IFNeuralGroup
        e_params = IFParams()
        i_params = IFParams()
    elif args.model == 'lif':
        model = LIFNeuralGroup
        e_params = LIFParams()
        i_params = LIFParams()
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

    # define the neural groups
    g1 = SensoryNeuralGroup(1, args.grid_size * args.grid_size, "inputs", 1, tki, e_params, field_shape=(args.grid_size, args.grid_size), viz_layer_pos=(0,0))
    g2 = model(1, 16, "exc", 2, tki, e_params, field_shape=(4,4), viz_layer_pos=(0,0))
    g3 = model(1, args.grid_size * 2, 'outputs', 3, tki, e_params, field_shape=(2, args.grid_size), viz_layer_pos=(0,0))

    # define the neural network
    nn = NeuralNetwork([g1, g2, g3], "bar_learner", tki)

    # set learning parameters
    lp = stdp_params
    lp.lr = 0.05

    # connect inputs layer to the excitatory layer
    nn.fully_connect("inputs", "exc", trainable=True, stdp_params=lp, s_type=args.stdp)
    # connect the excitatory layer to itself in the form of lateral inhibition
    nn.fully_connect("exc", "exc", w_i=1.0, trainable=False, skip_one_to_one=True, s_type='inhib')
    # connect exc layer to output layer
    nn.fully_connect("exc", "outputs", trainable=True, stdp_params=lp, s_type=args.stdp)
    # connect the output layer to itself in the form of lateral inhibition
    nn.fully_connect("outputs", "outputs", w_i=1.0, trainable=False, skip_one_to_one=True, s_type='inhib')

    # load data
    training_data = np.load('../datasets/bars/bar_data.npy')
    training_labels = np.load('../datasets/bars/bar_labels.npy')
    network_labels = ['0_0', '1_0', '2_0', '3_0', '4_0', '5_0', '6_0', '7_0', '8_0', '9_0', '10_0', '11_0', '12_0', '13_0', '14_0', '15_0',
                      '0_1', '1_1', '2_1', '3_1', '4_1', '5_1', '6_1', '7_1', '8_1', '9_1', '10_1', '11_1', '12_1', '13_1', '14_1', '15_1']
    
    # define order in which network will run
    run_order = ["inputs", "exc", "outputs"]

    # Define the run handler, this handles the processing pipeline
    nrh = NetworkRunHandler(nn,
                            training_data,
                            training_labels,
                            network_labels,
                            run_order,
                            output_encoding=NetworkRunHandler.MULTI_FIRST_OUTPUT_ENC,
                            enable_additive_frequency=True,
                            enable_subtractive_frequency=True,
                            base_input_frequency=100.0,
                            additive_frequency_boost=5.0,
                            additive_frequency_wait=50.0,
                            enable_noise_sampling=True,
                            noise_sampling_interval=3.0,
                            noise_probability=0.1,
                            normalize_on_weight_change=True,
                            normalize_on_start=True,
                            episodes=args.e,
                            save_at_end=False,
                            save_dir=None,
                            reset_on_process=True,
                            rewards=(-1.0, 1.0),
                            data_pre_processor=None,
                            training=True,
                            exposure_hits=1,
                            output_is_ftm=False,
                            ftm_supervised=False,
                            allow_multi_wrong=False)
    
    window = VizWindow(nrh, no_show_inhib=True)
    pyglet.app.run()

    # d = genbar(args.grid_size, args.grid_size)
    # last_exposure_step = 0
    # cummulative_spikes = 0
    # f_add = 0
    # for step in tki:
    #     if (step - last_exposure_step) * tki.dt() >= args.exposure * msec:
    #         if cummulative_spikes == 0:
    #             f_add += 5
    #         else:
    #             cummulative_spikes = 0
    #             last_exposure_step = step
    #             d = genbar(args.grid_size, args.grid_size)
    #             nn.reset()
    #             nn.normalize_weights()
         
    #     # inject spikes into sensory layer
    #     g1.run(poisson_train(d, tki.dt(), args.input_rate + f_add))

    #     # run all layers
    #     nn.run_order(["inputs", "exc"])

    #     cummulative_spikes += g2.spike_count.sum()
        
    #     sys.stdout.write("Current simulation time :: %.0f ms :: Num Spikes :: %g                  \r" % (step * tki.dt() / msec, cummulative_spikes))

    #     if step >= duration/tki.dt():
    #         break
    
    # print("\n\n")

    # -------------------------------------------------------
    img = form_full_weight_map(g2, "inputs", "exc", nn)
    plt.imshow(img)
    plt.title("Post-training weight matrix")
    plt.colorbar()
    plt.clim(0, img.max())
    plt.show()
    # -------------------------------------------------------
