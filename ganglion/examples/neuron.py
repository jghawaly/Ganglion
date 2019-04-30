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
    parser = argparse.ArgumentParser(description='Run a simulation of a single neuron.')
    parser.add_argument('model', type=str, help='the neuron model to evaluate')
    parser.add_argument('--duration', type=float, default=1000.0, help='duration of simulation (milliseconds)')
    parser.add_argument('--increment', type=float, default=0.1, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--input_current', type=float, default=1.0, help='input current (nanoamps)')
    parser.add_argument('--input_current_time', type=float, default=0.5, help='fraction of simulation duration in which input current is applied (0.0 < x < 1.0)')

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
    current_duration = args.input_current_time * duration

    n = model(np.ones((1,1), dtype=np.int), "Neuron", tki, params())
    n.tracked_vars = ["v_m", "i_syn"]

    for step in tki:
        if tki.tick_time() <= current_duration:
            # constant current input
            n.run(args.input_current * namp)
        else:
            n.run(0.0 * namp)
        sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

        if step >= duration/tki.dt():
            break

    v = [i[0] for i in n.v_m_track] 
    isyn = n.isyn_track

    times = np.arange(0,len(v), 1) * tki.dt() / msec

    plt.plot(times, v, 'k')
    plt.title("Voltage Track")
    plt.xlabel("Time (msec)")
    plt.ylabel("Membrane Potential (volt)")
    plt.show()

    plt.plot(times, isyn, 'k')
    plt.title("Synaptic Current Track")
    plt.xlabel("Time (msec)")
    plt.ylabel("Synaptic Current (amp)")
    plt.show()
