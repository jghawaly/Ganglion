import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, IFNeuralGroup, LIFNeuralGroup, FTLIFNeuralGroup, ExLIFNeuralGroup, AdExNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import IFParams, LIFParams, FTLIFParams, ExLIFParams, AdExParams, PairSTDPParams, TripletSTDPParams
from units import *
from utils import poisson_train
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import argparse

"""
This example demonstrates the ability of STDP to extract statistically-regular patterns in an input while ignoring noise
"""

class Bar:
    def __init__(self, grid_size):
        """
        initialize the bar grid
        @param grid_size: size of the square grid
        """
        self.grid = np.zeros((grid_size, grid_size), dtype=np.float)
        self.bar_col = np.random.randint(0, grid_size)
        self.grid[:, self.bar_col] = 1.0
    
    def sample(self, noise_prob):
        """
        return the grid with added noise
        @param noise_prob: probability of each pixel containing salt and pepper noise
        """
        grid = self.grid.copy()

        # generate noise
        for idx, _ in np.ndenumerate(grid):
            if random.random() <= noise_prob:
                grid[idx] = 1.0
        
        return grid/grid.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation of a neural network that learns to detect bars in an image.')
    parser.add_argument('--model', type=str, default='ftlif', help='the neuron model to evaluate, if, lif, ftlif, exlif, adex')
    parser.add_argument('--duration', type=float, default=100000.0, help='duration of simulation (milliseconds)')
    parser.add_argument('--increment', type=float, default=0.5, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--input_rate', type=float, default=64.0, help='maximum firing rate of input sensory neurons (Hz)')
    parser.add_argument('--grid_size', type=int, default=16, help='length and width of square grid that bars are generated on')
    parser.add_argument('--stdp', type=str, default='triplet', help='form of stdp to use, can be pair or triplet')
    parser.add_argument('--noise_prob', type=float, default=0.1, help='probability of each pixel containing noise')
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate of STDP')

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
        raise RuntimeError("%s is not a valid neuron model, must be if, lif, ftlif, exlif, or adex." % args.model)
    
    if args.stdp == 'pair':
        stdp_params = PairSTDPParams
    elif args.stdp == 'triplet':
        stdp_params = TripletSTDPParams
    else:
        raise RuntimeError("%s is not a valid stdp model, must be pair or triplet." % args.stdp)

    start = time.time()
    tki = TimeKeeperIterator(timeunit=args.increment*msec)
    duration = args.duration * msec

    inhib_layer_params = LIFParams()

    exc_layer_params = params()

    g1 = SensoryNeuralGroup(np.ones(args.grid_size * args.grid_size, dtype=np.int), "inputs", tki, exc_layer_params, field_shape=(args.grid_size, args.grid_size))
    g2 = model(np.ones(1, dtype=np.int), "exc", tki, exc_layer_params)

    nn = NeuralNetwork([g1, g2], "stdp", tki)
    lp = stdp_params()
    lp.lr = args.lr

    nn.fully_connect("inputs", "exc", trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type=args.stdp)

    bar = Bar(args.grid_size)

    for step in tki:
        # inject spikes into sensory layer
        g1.run(poisson_train(bar.sample(args.noise_prob), tki.dt(), args.input_rate))

        # run all layers
        nn.run_order(["inputs", "exc"])
        
        sys.stdout.write("Current simulation time :: %.0f ms                  \r" % (step * tki.dt() / msec))

        if step >= duration/tki.dt():
            break
    
    print("\n\n")

    # -------------------------------------------------------
    img = np.reshape(nn.get_w_between_g_and_g('inputs', 'exc'), (args.grid_size, args.grid_size))
    plt.imshow(img)
    plt.title("Post-training weight matrix")
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()
    # -------------------------------------------------------