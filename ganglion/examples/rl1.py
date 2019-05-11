import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import LIFNeuralGroup, SensoryNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import LIFParams, DASTDPParams
from units import *
from utils import poisson_train
import numpy as np
import numpy.random as nprand
import time
import random
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt


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
    def __init__(self, grid_shape=(8, 4), speed=4.0, same_col=False, reward_scheme=(-1.0, 1.0, 0.0)):
        self.speed = speed
        self.step_duration = 1.0 / speed / msec
        self.shape = grid_shape
        self.same_col = same_col
        self.reward_scheme = reward_scheme

        col_ap = nprand.randint(0, self.shape[1])
        col_bp = col_ap if self.same_col else nprand.randint(0, self.shape[1])
        self.ap = (self.shape[0]-1, col_ap)
        self.bp = (0, col_bp)
    
    def step(self, action):
        last_agent_position = self.ap
        if action == 0:
            self.ap = (self.ap[0], self.ap[1]-1)
        elif action == 1:
            self.ap = (self.ap[0], self.ap[1]+1)
        else:
            pass
        
        if self.ap[1] == self.shape[1] or self.ap[1] < 0:
            self.ap = last_agent_position

        self.bp = (self.bp[0]+1, self.bp[1])

        if self.bp == self.ap:
            reward = self.reward_scheme[0]
        elif self.bp[0] == self.shape[0] - 1:
            reward = self.reward_scheme[1]
        else:
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
        state = np.zeros(2 * self.shape[1] - 1, dtype=np.float)
        diff = self.ap[1] - self.bp[1]

        state[diff + self.shape[1] - 1] = 1.0

        return state
    
    def reset(self):
        col_ap = nprand.randint(0, self.shape[1])
        col_bp = col_ap if self.same_col else nprand.randint(0, self.shape[1])
        self.ap = (self.shape[0]-1, col_ap)
        self.bp = (0, col_bp)

if __name__ == "__main__":
    game = DodgeWorld(same_col=False, reward_scheme=(1.0, -1.0, 0.0))
    exposure = game.step_duration
 
    input_rate = 64.0

    tki = TimeKeeperIterator(timeunit=0.1*msec)
    num_episodes = 500

    exc_layer_params = LIFParams()

    g1 = SensoryNeuralGroup(np.ones(2 * game.shape[1] - 1, dtype=np.int), "1", tki, exc_layer_params)
    g2 = LIFNeuralGroup(np.ones(5, dtype=np.int), "2", tki, exc_layer_params)
    g3 = LIFNeuralGroup(np.ones(3, dtype=np.int), "3", tki, exc_layer_params)

    nn = NeuralNetwork([g1, g2, g3], "dodge ball player", tki)
    lp = DASTDPParams()
    lp.lr_pre = 0.001
    lp.lr_post = 0.001

    nn.fully_connect("1", "2", trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
    nn.fully_connect("2", "3", trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')

    last_exposure_step = 1
    cummulative_spikes = np.zeros(g3.shape)
    state = game.get_state()
    i = 0

    print(game.get_pixel_state())
    scores = []

    for step in tki:
        if (step - last_exposure_step) * tki.dt() >= exposure * msec:
            last_exposure_step = step
            action = cummulative_spikes.argmax()
            reward, state = game.step(action)

            if reward != 0.0:
                nn.dopamine_puff(reward)
                game.reset()
                state = game.get_state()
                i += 1
                scores.append(reward)

            cummulative_spikes.fill(0)

            nn.reset()
            print(game.get_pixel_state())

        # inject spikes into sensory layer
        g1.run(poisson_train(state, tki.dt(), input_rate))

        # run all layers
        nn.run_order(["1", "2", "3"])

        cummulative_spikes += g3.spike_count
        
        sys.stdout.write("Current simulation time :: %.0f ms :: Num Spikes :: %g                  \r" % (step * tki.dt() / msec, cummulative_spikes.sum()))

        if i == num_episodes:
            break
    
    print("\n\n")
    nn.save_w("g1_g2.npy", '1', '2')
    nn.save_w("g2_g3.npy", '2', '3')
    plt.plot(scores)
    plt.show()
# if __name__ == "__main__":
#     game = DodgeWorld(same_col=False, reward_scheme=(1.0, -1.0, 0.0))
#     exposure = game.step_duration
 
#     input_rate = 64.0

#     tki = TimeKeeperIterator(timeunit=0.1*msec)
#     num_episodes = 50

#     exc_layer_params = LIFParams()

#     g1 = SensoryNeuralGroup(np.ones(game.shape[0] * game.shape[1], dtype=np.int), "1", tki, exc_layer_params, field_shape=game.shape)
#     g2 = LIFNeuralGroup(np.ones(5, dtype=np.int), "2", tki, exc_layer_params)
#     g3 = LIFNeuralGroup(np.ones(3, dtype=np.int), "3", tki, exc_layer_params)

#     nn = NeuralNetwork([g1, g2, g3], "dodge ball player", tki)
#     lp = DASTDPParams()

#     nn.fully_connect("1", "2", trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')
#     nn.fully_connect("2", "3", trainable=True, stdp_params=lp, minw=0.1, maxw=0.5, s_type='da')

#     last_exposure_step = 1
#     cummulative_spikes = np.zeros(g3.shape)
#     state = game.get_pixel_state()
#     i = 0

#     print(state)

#     for step in tki:
#         if (step - last_exposure_step) * tki.dt() >= exposure * msec:
#             last_exposure_step = step
#             action = cummulative_spikes.argmax()
#             reward, state = game.step(action)

#             if reward != 0.0:
#                 nn.dopamine_puff(reward)
#                 game.reset()
#                 state = game.get_pixel_state()
#                 i += 1

#             cummulative_spikes.fill(0)

#             nn.reset()
#             print(state)

#         # inject spikes into sensory layer
#         g1.run(poisson_train(state, tki.dt(), input_rate))

#         # run all layers
#         nn.run_order(["1", "2", "3"])

#         cummulative_spikes += g3.spike_count
        
#         sys.stdout.write("Current simulation time :: %.0f ms :: Num Spikes :: %g                  \r" % (step * tki.dt() / msec, cummulative_spikes.sum()))

#         if i == num_episodes:
#             break
    
#     print("\n\n")
#     nn.save_w("g1_g2.npy", '1', '2')
#     nn.save_w("g2_g3.npy", '2', '3')
