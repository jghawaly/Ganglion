import sys
sys.path.append("../base")

from timekeeper import TimeKeeperIterator
from NeuralGroup import *
from NeuralNetwork import NeuralNetwork
from parameters import *
from units import *
from utils import poisson_train, calculate_phi
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate an F-I gain curve for the given model.')
    parser.add_argument('--model', type=str, default='amlif', help='the neuron model to evaluate, if, lif, ftlif, exlif, adex, amlif, or hslif')
    parser.add_argument('--duration', type=float, default=10000.0, help='duration of simulation (milliseconds)')
    parser.add_argument('--increment', type=float, default=0.1, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--imin', type=float, default=0.0, help='Input current to start at (pA)')
    parser.add_argument('--imax', type=float, default=50.0, help='Input current to stop at (pA)')
    parser.add_argument('--itime', type=float, default=10000.0, help='Time to integrate spikes over each input current interval (milliseconds)')
    parser.add_argument('--intervals', type=int, default=100, help='Number of discrete synaptic current intervals between imin and imax')
    args = parser.parse_args()

    # setup timing
    tki = TimeKeeperIterator(timeunit=args.increment*msec)
    integration_time = args.itime * msec
    duration = integration_time * args.intervals
    i_syn = np.linspace(args.imin, args.imax, args.intervals)

    # chooses neuron model and parameters
    if args.model == 'if':
        model = IFNeuralGroup
        params = IFParams()
    elif args.model == 'lif':
        model = LIFNeuralGroup
        params = LIFParams()
    elif args.model == 'amlif':
        model = AMLIFNeuralGroup
        params = AMLIFParams()
        # params.sa = 0.005
        # params.ca_tau = 300 * msec
    elif args.model == 'ftlif':
        model = FTLIFNeuralGroup
        params = FTLIFParams()
    elif args.model == 'exlif':
        model = ExLIFNeuralGroup
        params = ExLIFParams()
    elif args.model == 'adex':
        model = AdExNeuralGroup
        params = AdExParams()
    elif args.model == 'hslif':
        model = HSLIFNeuralGroup
        params = HSLIFParams()
        params.phi = calculate_phi(args.target_frequency, tki)
    else:
        raise RuntimeError("%s is not a valid neuron model, must be if, lif, ftlif, exlif, adex, or hslif." % args.model)

    params.refractory_period = 0

    n = model(1, 1, "Neuron", 1, tki, params)

    rates = np.zeros(i_syn.shape, dtype=int)

    i = 1
    for step in tki:
        sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

        # constant current input
        n.run(i_syn[i-1] * pamp)

        rates[i-1] += n.spike_count[0]

        if step >= args.intervals * integration_time / tki.dt():
            break

        if step >= i * integration_time / tki.dt():
            i += 1
            rates[i-1] = rates[i-1] / integration_time
            n.reset()
        

    plt.plot(i_syn, rates, 'k')
    plt.xlabel("$I_{syn}$ (pA)", fontsize=16)
    plt.ylabel("AP Frequency (Hz)", fontsize=16)
    plt.title("F-I Curve", fontsize=18)
    plt.show()
