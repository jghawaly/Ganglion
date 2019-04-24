import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, IFNeuralGroup, LIFNeuralGroup, FTLIFNeuralGroup, ExLIFNeuralGroup, AdExNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import IFParams, LIFParams, FTLIFParams, ExLIFParams, AdExParams
from units import *
from utils import poisson_train
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a simulation of a neural network with 3 excitatory layers. Network is untrainable with weights initialized to 0.5.')
    parser.add_argument('model', type=str, help='the neuron model to evaluate')
    parser.add_argument('--duration', type=float, default=1000.0, help='duration of simulation (milliseconds)')
    parser.add_argument('--increment', type=float, default=0.1, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--input_rate', type=float, default=64.0, help='firing rate of input sensory neurons (Hz)')
    parser.add_argument('--num_input_neurons', type=int, default=3, help='number of neurons in the input layer')
    parser.add_argument('--num_hidden_neurons', type=int, default=3, help='number of neurons in the hidden layer')
    parser.add_argument('--num_output_neurons', type=int, default=3, help='number of neurons in the output layer')

    args = parser.parse_args()

    if args.model == 'if':
        model = IFNeuralGroup
        params = IFParams
    elif args.model == 'lif':
        model = LIFNeuralGroup
        params = LIFParams
    elif args.model == 'ftlif':
        model = FTLIFNeuralGroup
        params = FTLIFParams
    elif args.model == 'exlif':
        model = ExLIFNeuralGroup
        params = ExLIFParams
    elif args.model == 'adex':
        model = AdExNeuralGroup
        params = AdExParams
    else:
        raise RuntimeError("%s is not a valid neuron model, must be if, lif, ftlif, exlif, or adex.")
    
    tki = TimeKeeperIterator(timeunit=args.increment*msec)
    duration = args.duration * msec
    g1 = SensoryNeuralGroup(np.ones(args.num_input_neurons, dtype=np.int), "input", tki, params())
    g2 = model(np.ones(args.num_hidden_neurons, dtype=np.int), "hidden", tki, params())
    g3 = model(np.ones(args.num_output_neurons, dtype=np.int), "output", tki, params())
    g3.tracked_vars = ["v_m", "i_syn"]

    nn = NeuralNetwork([g1, g2, g3], "blah", tki)
    nn.fully_connect("input", "hidden", w_i=0.5, trainable=False)
    nn.fully_connect("hidden", "output", w_i=0.5, trainable=False)
    
    for step in tki:
        # inject spikes into sensory layer
        g1.run(poisson_train(np.ones(args.num_input_neurons, dtype=float), tki.dt(), args.input_rate))
        # run all layers
        nn.run_order(["input", "hidden", "output"])
        
        sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

        if step >= duration/tki.dt():
            break

    v = [i[0] for i in g3.v_m_track] 
    isyn = [i[0] for i in g3.isyn_track] 

    times = np.arange(0,len(v), 1) * tki.dt() / msec

    plt.plot(times, v)
    plt.title("Voltage Track")
    plt.xlabel("Time (msec)")
    plt.ylabel("Membrane Potential (volt)")
    plt.show()

    plt.plot(times, isyn)
    plt.title("Synaptic Current Track")
    plt.xlabel("Time (msec)")
    plt.ylabel("Synaptic Current (amp)")
    plt.show()

