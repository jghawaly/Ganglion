import sys
sys.path.append("../base")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, IFNeuralGroup, LIFNeuralGroup, FTLIFNeuralGroup, ExLIFNeuralGroup, AdExNeuralGroup, HSLIFNeuralGroup, AMLIFNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import IFParams, LIFParams, FTLIFParams, ExLIFParams, AdExParams, HSLIFParams, AMLIFParams
from units import *
from utils import poisson_train, calculate_phi
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a simulation of a single neuron.')
    parser.add_argument('model', type=str, help='the neuron model to evaluate')
    parser.add_argument('--duration', type=float, default=1000.0, help='duration of simulation (milliseconds)')
    parser.add_argument('--increment', type=float, default=0.01, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--input_current', type=float, default=1.0, help='input current (nanoamps)')
    parser.add_argument('--input_current_time', type=float, default=0.5, help='fraction of simulation duration in which input current is applied (0.0 < x < 1.0)')
    parser.add_argument('--target_frequency', type=float, default=10, help='target frequency in Hz of neuron (only applicable to HSLIF neurons.')

    args = parser.parse_args()

    # setup timing
    tki = TimeKeeperIterator(timeunit=args.increment*msec)
    duration = args.duration * msec
    current_duration = args.input_current_time * duration

    # number of plots to make
    num_plots = 2

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
        params.sa = 0.005
        params.ca_tau = 300 * msec
        num_plots += 1
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
        num_plots += 1
    else:
        raise RuntimeError("%s is not a valid neuron model, must be if, lif, ftlif, exlif, adex, or hslif." % args.model)

    n = model(1, 1, "Neuron", 1, tki, params)
    n.tracked_vars = ["v_m", "i_syn", "v_thr", "m"]

    for step in tki:
        if tki.tick_time() <= current_duration:
            # constant current input
            n.run(args.input_current * namp)
        else:
            n.run(0.0 * namp)
        sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

        if step >= duration/tki.dt():
            break
    
    # get tracked data
    v = [i[0] for i in n.v_m_track] 
    isyn = n.isyn_track

    # time axis data
    times = np.arange(0,len(v), 1) * tki.dt() / msec

    fig, ax = plt.subplots(ncols=1, nrows=num_plots, sharex=True)

    ax[0].plot(times, v, 'k')
    ax[0].set_title("Membrane Potential Track", fontsize=18)
    ax[0].set_ylabel("$V_m$ (volt)", fontsize=16)

    ax[1].plot(times, isyn, 'k')
    ax[1].set_title("Synaptic Current Track", fontsize=18)
    ax[1].set_ylabel("$I_{\text{syn}}$ (amp)", fontsize=16)

    if args.model == 'hslif':
        v_thr = [i[0] for i in n.v_thr_track] 
        ax[2].plot(times, v_thr, 'k')
        ax[2].set_title("Threshold Potential Track", fontsize=18)
        ax[2].set_ylabel("$V_{\text{thr}}$ (volt)", fontsize=16)
    
    if args.model == 'amlif':
        m = [i[0] for i in n.m_track] 
        ax[2].plot(times, m, 'k')
        ax[2].set_title("Adaptive Conductance Multiplier", fontsize=18)
        ax[2].set_ylabel("M", fontsize=16)
    
    ax[-1].set_xlabel("Time (msec)", fontsize=16)
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.show()
