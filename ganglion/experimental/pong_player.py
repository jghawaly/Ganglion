import sys
sys.path.append("../base")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, AMLIFNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import IFParams, AMLIFParams, DASTDPParams, SynapseParams
from units import *
from utils import poisson_train, calculate_phi
import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse


# This is a work in progress !!!!!

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation of a neural network that learns to play Pong.')
    parser.add_argument('--increment', type=float, default=0.25, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--e', type=int, default=160, help='number of episodes to run')
    args = parser.parse_args()

    # create timekeeper
    tki = TimeKeeperIterator(timeunit=args.increment * msec)

    # SENSORY LAYER ########################################################
    sp = IFParams()
    sp.gbar_e = 50 * nsiem
    g1 = SensoryNeuralGroup(1, 128, "ram", 1, tki, params=sp)

    # HIDDEN LAYER #########################################################
    hp = AMLIFParams()
    hp.gbar_e = 50 * nsiem
    g2 = AMLIFNeuralGroup(1, 10, "hidden", 2, tki, params=hp)

    # DECISION LAYER #######################################################
    dp = AMLIFParams()
    g3 = AMLIFNeuralGroup(1, 3, "decision", 3, tki, params=dp)

    # NETWORK ##############################################################
    nn = NeuralNetwork([g1, g2, g3], "pong_player", tki)

    # CONNECTIONS ##########################################################
    dap = DASTDPParams()
    dap.lr = 0.1
    # feedforward excitation
    nn.fully_connect("ram", "hidden", trainable=True, stdp_params=dap, syn_params=None, minw=0.01, maxw=0.9, s_type='da')
    # feedforward excitation
    nn.fully_connect("hidden", "decision", trainable=True, stdp_params=dap, syn_params=None, minw=0.01, maxw=0.9, s_type='da')
    # lateral inhibition to encourage sparsity in hidden layer
    nn.fully_connect("hidden", "hidden", trainable=False, w_i=1.0, s_type='inhib')
    # lateral inhibition to encourage only single decision in output layer
    nn.fully_connect("decision", "decision", trainable=False, w_i=1.0, s_type='inhib')

    nn.normalize_weights()


    # NOOP = 0, UP = 2, DOWN = 3
    actions = [0, 2, 3]

    # Intialize Pong game
    env = gym.make("Pong-ram-v0")

    # Begin first episode of game
    observation = env.reset()

    # current episode
    e_i = 0
    # this will contain the action
    action = 0
    # start rendering
    env.render()
    # initial observation
    observation, _, _, _ = env.step(action)
    while e_i < args.e:
        env.render()
        # RUN NETWORK ##########################################################
        decision_made = False
        fadd = 0
        etime = 0
        while not decision_made:
            tki.__next__()
            etime += tki.dt()
            print(tki.tick_time() / msec, end="\r")
            # inject spikes into sensory layer
            g1.run(poisson_train(observation/256, tki.dt(), (1000 + fadd)/256).reshape((128, 1)))
            # run network
            nn.run_order(['ram', 'hidden', 'decision'])
            # tally output num_spikes
            num_spikes = g3.spike_count.sum()
            if num_spikes == 1:
                action = actions[np.argmax(g3.spike_count)]
                decision_made = True
            elif num_spikes == 0:
                if etime > 20*msec:
                    fadd += 100
                    etime = 0
            elif g3.spike_count[0]==g3.spike_count[1]==g3.spike_count[2]:
                pass
            else:
                action = actions[np.argmax(g3.spike_count)]
                decision_made = True
        fadd = 0
        # run one step
        observation, reward, done, info = env.step(action)
        if reward != 0:
            nn.dopamine_puff(reward, actions=action)
            nn.normalize_weights()
        #self.nn.dopamine_puff(reward, actions=action)

        # if the episode is over, reset the environment
        if done:
            env.reset()
            nn.reset()
            observation, _, _, _ = env.step(0)
            e_i += 1