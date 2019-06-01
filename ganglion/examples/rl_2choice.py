import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import FTLIFNeuralGroup, SensoryNeuralGroup, LIFNeuralGroup, AdExNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import FTLIFParams, DASTDPParams, LIFParams, AdExParams
from units import *
from utils import poisson_train
import numpy as np
import numpy.random as nprand
import time
import random
import matplotlib.pyplot as plt


"""
This is a work in progress
"""

if __name__ == "__main__":
    exposure = 200.0
    input_rate = 360.
    num_episodes = 1000

    # inputs and desired outputs
    inputs = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]
    outputs = [1.0, 0.0]

    tki = TimeKeeperIterator(timeunit=0.5*msec)

    n_params = LIFParams()
    n_params.gbar_e = 20.0 * nsiem
    n_params.gbar_i = 300.0 * nsiem
    n_params.tao_m = 150 * msec

    lateral_params = LIFParams()
    lateral_params.gbar_i = 300.0 * nsiem
    lateral_params.tao_m = 150 * msec
    lateral_params.v_thr = -70.0 * mvolt
    epsilon = 0.8

    g1 = SensoryNeuralGroup(np.ones(2, dtype=np.int), "1", tki, n_params)
    g2 = LIFNeuralGroup(np.ones(2, dtype=np.int), "2", tki, n_params)
    g2i = LIFNeuralGroup(np.zeros(2, dtype=np.int), "2i", tki, n_params)
    g2il = LIFNeuralGroup(np.zeros(2, dtype=np.int), '2il', tki, lateral_params)
    g3 = LIFNeuralGroup(np.ones(2, dtype=np.int), "3", tki, n_params)
    g3i = LIFNeuralGroup(np.zeros(2, dtype=np.int), "3i", tki,n_params)
    g3il = LIFNeuralGroup(np.zeros(2, dtype=np.int), '3il', tki, lateral_params)
    nn = NeuralNetwork([g1, g2, g2il, g3, g3il], "SIMPLE", tki)
    
    lp = DASTDPParams()
    lp.lr = 0.01

    motor_params = DASTDPParams()
    motor_params.motor_excitability = False
    motor_params.lr = 0.01

    nn.fully_connect("1", "2", trainable=True, stdp_params=lp, minw=0.4, maxw=0.7, s_type='da')
    nn.fully_connect("2", "3", trainable=True, stdp_params=motor_params, minw=0.4, maxw=0.7, s_type='da')

    nn.one_to_one_connect('2', '2il', trainable=False, w_i=1.0)
    nn.fully_connect('2il', '2', trainable=False, w_i=1.0, skip_one_to_one=True)
    nn.one_to_one_connect('3', '3il', trainable=False, w_i=1.0)
    nn.fully_connect('3il', '3', trainable=False, w_i=1.0, skip_one_to_one=True)

    nn.normalize_weights()

    last_exposure_step = 1
    cummulative_spikes = np.zeros(g3.shape)
    i = 0

    state_index = nprand.randint(0, len(inputs))
    scores = []
    ft_add = 0

    for step in tki:
        if (step - last_exposure_step) * tki.dt() >= exposure * msec:
            last_exposure_step = step
            if cummulative_spikes.sum() > 0:
                dice = random.random()
                if dice <= epsilon:
                    action = np.random.choice(np.arange(0, cummulative_spikes.shape[0], 1), p=cummulative_spikes/cummulative_spikes.sum())
                else:
                    if cummulative_spikes[0] == cummulative_spikes[1]:
                        action = random.randint(0, 1)
                    else:
                        action = cummulative_spikes.argmax()
                # print(nn.get_w_between_g_and_g('2', '3'))
                # print(cummulative_spikes)
                # print(inputs[state_index])
                # print(action)
                
                reward = 1.0 if action == outputs[state_index] else -1.0
                state_index = nprand.randint(0, len(inputs))

                nn.dopamine_puff(reward, actions=action)
                # print(nn.get_w_between_g_and_g('2', '3'))
                # print()
                i += 1
                scores.append(reward)

                cummulative_spikes.fill(0)

                nn.reset()
                nn.normalize_weights()
                ft_add = 0
            else:
                ft_add += 5

        # inject spikes into sensory layer
        g1.run(poisson_train(inputs[state_index], tki.dt(), input_rate + ft_add))

        # run all layers
        nn.run_order(["1", "2", "2il", "3", "3il"])

        cummulative_spikes += g3.spike_count
        
        sys.stdout.write("Current simulation time :: %.0f ms :: Current episode :: %g :: Num Spikes :: %g                  \r" % (step * tki.dt() / msec, i, cummulative_spikes.sum()))

        if i == num_episodes:
            break
    
    print("\n\n")
    # nn.save_w("g1_g2.npy", '1', '2')
    # nn.save_w("g2_g3.npy", '2', '3')
    plt.plot(scores)
    plt.show()
