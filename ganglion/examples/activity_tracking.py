import sys
sys.path.append("../base")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, IFNeuralGroup, LIFNeuralGroup, FTLIFNeuralGroup, ExLIFNeuralGroup, AdExNeuralGroup, HSLIFNeuralGroup, AMLIFNeuralGroup
from NeuralNetwork import NeuralNetwork
from NetworkRunHandler import NetworkRunHandler
from parameters import IFParams, LIFParams, FTLIFParams, ExLIFParams, AdExParams, PairSTDPParams, TripletSTDPParams, HSLIFParams, AMLIFParams
from units import *
from utils import poisson_train, calculate_phi
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Track the activity of a neural network.')
    parser.add_argument('--model', type=str, default='amlif', help='the neuron model to evaluate, if, lif, ftlif, exlif, adex, or hslif')
    parser.add_argument('--duration', type=float, default=10000.0, help='duration of simulation (milliseconds)')
    parser.add_argument('--increment', type=float, default=0.4, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--input_rate', type=float, default=2000.0, help='maximum firing rate of input sensory neurons (Hz)')

    args = parser.parse_args()
    # Define the time keeper and total duration of training
    tki = TimeKeeperIterator(timeunit=args.increment*msec)
    duration = args.duration * msec

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

    e_params.tao_m = 10.0 * msec
    e_params.gbar_e = 0.3 * nsiem
    e_params.gbar_i = 5.0 * e_params.gbar_e

    # define the neural groups
    g1 = SensoryNeuralGroup(1, 64, "inputs", 1, tki, e_params, field_shape=(8, 8), viz_layer_pos=(0,0))
    g2 = model(1, 16, "exc", 2, tki, e_params, field_shape=(4,4), viz_layer_pos=(0,0))

    # define the neural network andt tell it to track the network's total activity
    nn = NeuralNetwork([g1, g2], "bar_learner", tki, track_activity=True, tracking_time=20*msec)

    # connect inputs layer to the excitatory layer
    nn.fully_connect("inputs", "exc", trainable=False, s_type='base', minw=0.1, maxw=0.3)
    # connect the excitatory layer to itself in the form of lateral inhibition
    for z in range(20):
        nn.fully_connect("exc", "exc", w_i=20.0, trainable=False, skip_one_to_one=True, s_type='inhib')

    # load data
    training_data = np.load('../datasets/bars/bar_data.npy')
    training_labels = np.load('../datasets/bars/bar_labels.npy')
    
    # define order in which network will run
    run_order = ["inputs", "exc"]

    # these will store time and activity
    t = []
    f = []

    for step in tki:
        t.append(step * tki.dt())

        # inject spikes into sensory layer
        g1.run(poisson_train(training_data[0], tki.dt(), args.input_rate))
        # run all layers
        nn.run_order(run_order)
        
        # show progress
        sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

        # get and store current frequency
        f.append(nn.current_activity())

        if step >= duration/tki.dt():
            break
    
    plt.plot(np.array(t) / msec, np.array(f), 'k')
    plt.xlabel("Time (msec)")
    plt.ylabel("Network Spike Frequency (Hz)")
    plt.show()
