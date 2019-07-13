import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, IFNeuralGroup, LIFNeuralGroup, FTLIFNeuralGroup, ExLIFNeuralGroup, AdExNeuralGroup, HSLIFNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import IFParams, LIFParams, FTLIFParams, ExLIFParams, AdExParams, DASTDPParams, HSLIFParams
from units import *
from utils import poisson_train, calculate_phi
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import argparse
import numpy.random as nprand


def parse_iris():
    data = []
    labels = []
    with open('../datasets/iris/iris.data', 'r') as f2r:
        for line in f2r:
            line = line.strip().split(',')
            data.append(np.array([float(line[0])/7.9, float(line[1])/4.4, float(line[2])/6.9, float(line[3])/2.5], dtype=np.float))
            labels.append(line[4])
    
    return np.array(data), np.array(labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation of a neural network that learns to detect bars in an image.')
    parser.add_argument('--model', type=str, default='hslif', help='the neuron model to evaluate, if, lif, ftlif, exlif, adex, or hslif')
    parser.add_argument('--increment', type=float, default=0.5, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--exposure', type=float, default=100.0, help='the duration of time that the network is exposed to each training example')
    parser.add_argument('--input_rate', type=float, default=64.0, help='maximum firing rate of input sensory neurons (Hz)')
    parser.add_argument('--target_frequency', type=float, default=50, help='target frequency in Hz of neuron (only applicable to HSLIF neurons')
    parser.add_argument('--episodes', type=int, default=500, help='number of episodes')
    parser.add_argument('--chi', type=float, default=0.8, help='a float value betwen 0 and 1 that is the probability of sampling the next action based on relative firing rates')

    # load the datasets and define output neuron classes
    network_labels = ['Iris-virginica', 'Iris-versicolor', 'Iris-setosa']  # output neuron classes
    data, labels = parse_iris()
    max_vals = np.array([7.9, 4.4, 6.9, 2.5], dtype=np.float)

    args = parser.parse_args()
    tki = TimeKeeperIterator(timeunit=args.increment*msec)
    
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
    
    lp = DASTDPParams()
    lp.lr = 0.001

    start = time.time()
    i_params.v_thr = -60.0 * mvolt


    # Define neural groups
    g1 = SensoryNeuralGroup(1, 4, "g1", tki, e_params)

    g2 = model(1, 100, "g2", tki, e_params)
    g2i = LIFNeuralGroup(0, 100, "g2i", tki, i_params)

    g3 = model(1, 3, "g3", tki, e_params)
    g3i = LIFNeuralGroup(0, 3, "g3i", tki, i_params)


    nn = NeuralNetwork([g1, g2, g2i, g3, g3i], "IRIS", tki)

    # excitatory feed-forward
    nn.fully_connect("g1", "g2", trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.fully_connect("g2", "g3", trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')

    # inhibitory lateral feedback
    nn.one_to_one_connect("g2", "g2i", w_i=1.0, trainable=False)
    nn.fully_connect("g2i", "g2", w_i=1.0, trainable=False, skip_one_to_one=True)
    nn.one_to_one_connect("g3", "g3i", w_i=1.0, trainable=False)
    nn.fully_connect("g3i", "g3", w_i=1.0, trainable=False, skip_one_to_one=True)

    last_exposure_step = 1  # timestep of last exposure
    cummulative_spikes = np.zeros(g3.shape)  # initialize the cummulative spikes matrix
    state_index = nprand.randint(0, data.shape[0])  # initialize the first state to sample
    i = 0  # current episode
    scores = []  # list of scores
    ft_add = 0  # additive frequency

    last_result = "       "

    for step in tki:
        if (step - last_exposure_step) * tki.dt() >= args.exposure * msec:
            last_exposure_step = step
            if cummulative_spikes.sum() > 0:
                dice = random.random()
                if dice <= args.chi:
                    action = np.random.choice(np.arange(0, cummulative_spikes.shape[0], 1), p=cummulative_spikes/cummulative_spikes.sum())
                else:
                    if cummulative_spikes[0] == cummulative_spikes[1]:
                        action = random.randint(0, 2)
                    else:
                        action = cummulative_spikes.argmax()
                
                reward = 1.0 if network_labels[action] == labels[state_index] else -1.0
                last_result = "CORRECT" if reward > 0.0 else "WRONG  "
                state_index = nprand.randint(0, data.shape[0])

                nn.dopamine_puff(reward, actions=action)
                
                i += 1
                scores.append(reward)

                cummulative_spikes.fill(0)

                nn.reset()
                nn.normalize_weights()
                ft_add = 0
            else:
                ft_add += 5

        # inject spikes into sensory layer
        g1.run(poisson_train(data[state_index], tki.dt(), args.input_rate + ft_add))

        # run all layers
        nn.run_order(["g1", "g2", "g2i", "g3", "g3i"])

        cummulative_spikes += g3.spike_count
        
        sys.stdout.write("Current simulation time :: %.0f ms :: Current episode :: %g :: Last Result :: %s :: Num Spikes :: %g                  \r" % (step * tki.dt() / msec, i, last_result, cummulative_spikes.sum()))

        if i == args.episodes:
            break
    
    print("\n\n")
    # nn.save_w("g1_g2.npy", '1', '2')
    # nn.save_w("g2_g3.npy", '2', '3')
    plt.plot(scores)
    plt.show()
