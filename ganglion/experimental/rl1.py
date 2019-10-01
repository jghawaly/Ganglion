import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import LIFNeuralGroup, SensoryNeuralGroup, HSLIFNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import LIFParams, DASTDPParams, HSLIFParams
from units import *
from utils import poisson_train, calculate_phi
import numpy as np
import numpy.random as nprand
import time
import random
import matplotlib.pyplot as plt
import cv2


"""
This is a work in progress
"""

class GridWorld:
    def __init__(self, shape, num_obstacles):
        self.shape = shape

        # generate the obstacle map
        self.obstacle_map = np.zeros(shape, dtype=np.float)
        for i in range(num_obstacles):
            self.obstacle_map[nprand.randint(0, shape[0]), nprand.randint(1, shape[1]-1)] = 1.0
        
        # generate the player agent starting position and put it on a map
        self.position = (nprand.randint(0, shape[0]), 0)
        self.position_map = np.zeros(shape, dtype=np.float)
        self.position_map[self.position] = 1.0

        # generate the "candy" position on the map
        self.meal_map = np.zeros(shape, dtype=np.float)
        self.meal_map[nprand.randint(0, shape[0]), shape[1]-1] = 1.0
    
    def move_agent(self, direction):
        last_position = self.position
        # Go up
        if direction == 0:
            new_row = np.clip(self.position[0]-1, 0, self.shape[0]-1)
            new_col = self.position[1]
        # Move down
        elif direction == 1:
            new_row = self.position[0]+1
            new_col = self.position[1]
        # Move left
        elif direction == 2:
            new_row = self.position[0]
            new_col = self.position[1]-1
        # Move right
        elif direction == 3:
            new_row = self.position[0]
            new_col = self.position[1]+1
        # stay in position
        elif direction == 4:
            new_row = self.position[0]
            new_col = self.position[1]
        else:
            raise ValueError("%g is not a valid movement direction." % direction)
        
        reward = 0.0
        
        # check if the agent went off of the top or bottom edge of the map
        if new_row < 0 or new_row == self.shape[0]:
            new_row = last_position[0]
            reward = -1.0
        # check if the agent went off of the left or right edge of the map
        if new_col < 0 or new_col == self.shape[1]:
            new_col = self.position[1]
            reward = -1
        
        self.position = (new_row, new_col)

        if reward == 0.0:
            if self.obstacle_map[self.position] > 0.0:
                self.position = (nprand.randint(0, self.shape[0]), 0)
                reward = -1.0
        
            if self.meal_map[self.position] > 0.0:
                self.position = (nprand.randint(0, self.shape[0]), 0)
                reward = 1.0

        self.position_map.fill(0.0)
        self.position_map[self.position] = 1.0

        return reward, self.position!=last_position
    
    def get_state(self):
        return self.obstacle_map, self.position_map, self.meal_map
    
    def get_state_rgb(self):
        state = np.dstack((self.obstacle_map, self.position_map, self.meal_map))
        return state


class DodgeWorld:
    def __init__(self, grid_shape=(8, 4), speed=4.0, same_col=False, reward_scheme=(-1.0, 1.0, 0.0, -1.0)):
        self.speed = speed
        self.step_duration = 1.0 / speed / msec
        self.shape = grid_shape
        self.state_shape = (2 * self.shape[1] - 1 + self.shape[1])
        self.same_col = same_col
        self.reward_scheme = reward_scheme

        col_ap = nprand.randint(0, self.shape[1])
        col_bp = col_ap if self.same_col else nprand.randint(0, self.shape[1])
        self.ap = (self.shape[0]-1, col_ap)
        self.bp = (0, col_bp)
    
    def step(self, action):
        last_agent_position = self.ap
        if action == 0:  # move left
            self.ap = (self.ap[0], self.ap[1]-1)
        elif action == 1:  # move right
            self.ap = (self.ap[0], self.ap[1]+1)
        else:  # don't move
            pass
        self.bp = (self.bp[0]+1, self.bp[1])

        if self.ap[1] == self.shape[1] or self.ap[1] < 0:  # went out of bounds
            self.ap = last_agent_position
            reward = self.reward_scheme[3]
        elif self.bp == self.ap:  # collision between ball and actor
            reward = self.reward_scheme[0]
        elif self.bp[0] == self.shape[0] - 1:  # ball got all the way down without being caught
            reward = self.reward_scheme[1]
        else:  # ball still travelling
            reward = self.reward_scheme[2]
        
        return reward, self.get_state()
    
    def get_pixel_state(self):
        state = np.zeros(self.shape)
        state[self.ap] = 1.0
        try:
            state[self.bp] = 1.0
        except IndexError:
            pass
        return state
    
    def get_state(self):
        state = np.zeros(2 * self.shape[1] - 1 + self.shape[1], dtype=np.float)
        diff = self.ap[1] - self.bp[1]

        state[diff + self.shape[1] - 1] = 1.0
        state[2 * self.shape[1] - 1 + self.ap[1]] = 1.0

        return state
    
    def reset(self):
        col_ap = nprand.randint(0, self.shape[1])
        col_bp = col_ap if self.same_col else nprand.randint(0, self.shape[1])
        self.ap = (self.shape[0]-1, col_ap)
        self.bp = (0, col_bp)


if __name__ == "__main__":
    game = DodgeWorld(same_col=False, reward_scheme=(1.0, -0.01, 0.0, -0.01), grid_shape=(8, 3))
    
    exposure = 500.0
    input_rate = 60.
    num_episodes = 1000

    tki = TimeKeeperIterator(timeunit=1.0*msec)

    n_params = HSLIFParams()
    n_params.gbar_e = 100.0 * nsiem
    n_params.tao_m = 150 * msec
    n_params.phi = calculate_phi(2.0, tki)

    lateral_params = HSLIFParams()
    lateral_params.gbar_i = 100.0 * nsiem
    lateral_params.tao_m = 150 * msec
    lateral_params.v_thr = -60.0 * mvolt
    lateral_params.phi = calculate_phi(4.0, tki)
    epsilon = 0.8

    i_params = LIFParams()
    i_params.v_thr = -65.0 * mvolt

    g1 = SensoryNeuralGroup(1, 8, "1", 1, tki, n_params)
    g2 = HSLIFNeuralGroup(1, 4, "2", 2, tki, n_params)
    g2il = LIFNeuralGroup(0, 4, '2il', 2, tki, i_params)
    g3 = HSLIFNeuralGroup(1, 3, "3", 3, tki, n_params)
    g3il = LIFNeuralGroup(0, 3, '3il', 3, tki, i_params)
    nn = NeuralNetwork([g1, g2, g2il, g3, g3il], "catch", tki)
    
    lp = DASTDPParams()
    lp.lr = 0.1
    lp.ab_et_tao = 0.5 * sec   
    lp.ba_et_tao = 0.25 * sec

    nn.fully_connect("1", "2", trainable=True, stdp_params=lp, minw=0.4, maxw=0.7, s_type='da')
    nn.fully_connect("2", "3", trainable=True, stdp_params=lp, minw=0.4, maxw=0.7, s_type='da')

    nn.one_to_one_connect('2', '2il', trainable=False, w_i=1.0)
    nn.fully_connect('2il', '2', trainable=False, w_i=1.0, skip_one_to_one=True)
    nn.one_to_one_connect('3', '3il', trainable=False, w_i=1.0)
    nn.fully_connect('3il', '3', trainable=False, w_i=1.0, skip_one_to_one=True)

    nn.normalize_weights()

    last_exposure_step = 1
    cummulative_spikes = np.zeros(g3.shape)
    i = 0

    state = game.get_state()
    scores = []
    ft_add = 0

    print(game.get_pixel_state())

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

                reward, state = game.step(action)

                if reward != 0:
                    print(nn.get_w_between_g_and_g('2', '3'))
                    nn.dopamine_puff(reward, actions=action)
                    i += 1
                    scores.append(reward)
                    nn.normalize_weights()
                    nn.reset()
                    game.reset()
                    state = game.get_state()
                    print(nn.get_w_between_g_and_g('2', '3'))
                    print()
                
                cummulative_spikes.fill(0)
                
                ft_add = 0

                print(game.get_pixel_state())
            else:
                ft_add += 5

        # inject spikes into sensory layer
        g1.run(poisson_train(state, tki.dt(), input_rate + ft_add))

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

# if __name__ == "__main__":
#     game = DodgeWorld(same_col=False, reward_scheme=(1.0, -0.01, 0.0, -0.01), grid_shape=(8, 3))
#     exposure = game.step_duration
 
#     input_rate = 64.0

#     tki = TimeKeeperIterator(timeunit=0.5*msec)
#     num_episodes = 10000

#     exc_layer_params = LIFParams()

#     # g1 = SensoryNeuralGroup(np.ones(2 * game.shape[1] - 1 + game.shape[1], dtype=np.int), "1", tki, exc_layer_params)
#     g1 = SensoryNeuralGroup(np.ones(game.state_shape, dtype=np.int), "1", tki, exc_layer_params)
#     g2 = LIFNeuralGroup(np.ones(5, dtype=np.int), "2", tki, exc_layer_params)
#     g3 = LIFNeuralGroup(np.ones(3, dtype=np.int), "3", tki, exc_layer_params)

#     nn = NeuralNetwork([g1, g2, g3], "dodge ball player", tki)
#     lp = DASTDPParams()
#     lp.lr = 0.01

#     nn.fully_connect("1", "2", trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
#     nn.fully_connect("2", "3", trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')

#     last_exposure_step = 1
#     cummulative_spikes = np.zeros(g3.shape)
#     state = game.get_state()
#     i = 0

#     print(game.get_pixel_state())
#     scores = []
#     ft_add = 0

#     for step in tki:
#         if (step - last_exposure_step) * tki.dt() >= exposure * msec:
#             last_exposure_step = step
#             if cummulative_spikes.sum() > 0:
#                 action = cummulative_spikes.argmax()
#                 reward, state = game.step(action)

#                 if reward != 0.0:
#                     nn.dopamine_puff(reward)
#                     game.reset()
#                     state = game.get_state()
#                     i += 1
#                     scores.append(reward)

#                 cummulative_spikes.fill(0)

#                 nn.reset()
#                 ft_add = 0
#                 print(game.get_pixel_state())
                
#             else:
#                 ft_add += 5

#         # inject spikes into sensory layer
#         g1.run(poisson_train(state, tki.dt(), input_rate + ft_add))

#         # run all layers
#         nn.run_order(["1", "2", "3"])

#         cummulative_spikes += g3.spike_count
        
#         sys.stdout.write("Current simulation time :: %.0f ms :: Num Spikes :: %g                  \r" % (step * tki.dt() / msec, cummulative_spikes.sum()))

#         if i == num_episodes:
#             break
    
#     print("\n\n")
#     # nn.save_w("g1_g2.npy", '1', '2')
#     # nn.save_w("g2_g3.npy", '2', '3')
#     plt.plot(scores)
#     plt.show()
