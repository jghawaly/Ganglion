import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import ExLIFNeuralGroup, SensoryNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import ExLIFParams, STDPParams
from units import *
from utils import poisson_train
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


if __name__ == "__main__":
    grid_size = (3,11)
    my_grid = GridWorld((3,11), 5)
    plt.imshow(my_grid.get_state_rgb())
    plt.show()
    
    tki = TimeKeeperIterator(timeunit=0.5*msec)
    duration = 10000.0 * msec

    inh_layer_params = ExLIFParams()
    inh_layer_params.gbar_i = 50.0 * nsiem
    inh_layer_params.tao_m = 50 * msec

    exc_layer_params = ExLIFParams()
    exc_layer_params.gbar_e = 50.0 * nsiem
    exc_layer_params.tao_m = 100 * msec

    g1_obstacles = SensoryNeuralGroup(np.ones(grid_size[0] * grid_size[1], dtype=np.int), "obstacle_inputs", tki, exc_layer_params, field_shape=grid_size)
    g1_meal = SensoryNeuralGroup(np.ones(grid_size[0] * grid_size[1], dtype=np.int), "meal_inputs", tki, exc_layer_params, field_shape=grid_size)
    g1_agent = SensoryNeuralGroup(np.ones(grid_size[0] * grid_size[1], dtype=np.int), "agent_inputs", tki, exc_layer_params, field_shape=grid_size)

    g2 = ExLIFNeuralGroup(np.ones(9, dtype=np.int), "al1", tki, exc_layer_params, field_shape=(1,9))
    g2i = ExLIFNeuralGroup(np.zeros(9, dtype=np.int), "al1i", tki, inh_layer_params)

    g3 = ExLIFNeuralGroup(np.ones(25, dtype=np.int), "al2", tki, exc_layer_params)

    g4 = ExLIFNeuralGroup(np.ones(5, dtype=np.int), "go", tki, exc_layer_params)
    g4i = ExLIFNeuralGroup(np.zeros(5, dtype=np.int), "nogo", tki, inh_layer_params)

    g5 = ExLIFNeuralGroup(np.ones(5, dtype=np.int), "action", tki, exc_layer_params)
    g5i = ExLIFNeuralGroup(np.zeros(5, dtype=np.int), "actioni", tki, inh_layer_params)

    dopamine_surge = SensoryNeuralGroup(np.ones(1, dtype=np.int), "dopamine_surge", tki, exc_layer_params)
    dopamine_dip = SensoryNeuralGroup(np.ones(1, dtype=np.int), "dopamine_dip", tki, exc_layer_params)

    g5.tracked_vars = ["spike"]

    nn = NeuralNetwork([g1_obstacles, g1_meal, g1_agent, g2, g2i, g3, g4, g4i, g5, g5i, dopamine_dip, dopamine_surge], "dopamine dingo", tki)

    lp = STDPParams()
    lp.lr = 0.005
    lp.a2_minus = 8.0e-3
    lp.a3_minus = 3e-4

    patch1 = np.ones((3,3))
    nn.convolve_connect("obstacle_inputs", "al1", patch1, 0, 1, trainable=True, stdp_params=lp, minw=0.05, maxw=0.9)
    nn.convolve_connect("meal_inputs", "al1", patch1, 0, 1, trainable=True, stdp_params=lp, minw=0.05, maxw=0.9)
    nn.convolve_connect("agent_inputs", "al1", patch1, 0, 1, trainable=True, stdp_params=lp, minw=0.05, maxw=0.9)

    nn.one_to_one_connect("al1", "al1i", trainable=False, w_i=1.0)
    nn.fully_connect("al1i", "al1", trainable=False, skip_one_to_one=True, w_i=1.0)

    nn.fully_connect("al1", "al2", trainable=True, stdp_params=lp, minw=0.05, maxw=0.9)

    nn.fully_connect("al2", "go", trainable=True, stdp_params=lp, minw=0.05, maxw=0.9)
    nn.fully_connect("al2", "nogo", trainable=True, stdp_params=lp, minw=0.05, maxw=0.9)

    nn.fully_connect("go", "action", trainable=True, stdp_params=lp, minw=0.05, maxw=0.9)
    nn.fully_connect("nogo", "action", trainable=True, stdp_params=lp, minw=0.05, maxw=0.9)

    nn.one_to_one_connect("action", "actioni", trainable=False, w_i=1.0)
    nn.fully_connect("actioni", "action", trainable=False, w_i=0.0)

    nn.fully_connect("dopamine_surge", "go", trainable=False, w_i=1.0)
    nn.fully_connect("dopamine_dip", "nogo", trainable=False, w_i=1.0)
    nn.fully_connect("dopamine_surge", "al1", trainable=False, w_i=1.0)
    nn.fully_connect("dopamine_dip", "al1i", trainable=False, w_i=1.0)

    lts = 0
    reward_type = 0
    for step in tki:
        if (step - lts)*tki.dt() >= 50*msec:
            if tki.tick_time() < 5000*msec:
                reward_type,change = my_grid.move_agent(nprand.randint(0, 5))
                print()
                print("exploration move")
                print()
                g5.spike_track = []
            else:
                spikes = np.array(g5.spike_track)
                rate = np.sum(spikes, axis=0) * 100.0 / np.sum(spikes)
                action = np.argmax(rate)
                if np.sum(spikes) == 0:
                    action = 4
                lts = step
                reward_type, change = my_grid.move_agent(action)
                g5.spike_track = []
                print()
                print(rate)
                print(action)
                
                if change:
                    plt.imshow(my_grid.get_state_rgb())
                    plt.show()

                print()

        

        # reward the network for 5 milliseconds 
        if (step - lts)*tki.dt() >= 5*msec:
            if reward_type == -1:
                dopamine_dip.run(poisson_train(np.ones(1, dtype=np.float), tki.dt(), 0))
                dopamine_surge.run(poisson_train(np.zeros(1, dtype=np.float), tki.dt(), 0))
            if reward_type == 1:
                dopamine_dip.run(poisson_train(np.zeros(1, dtype=np.float), tki.dt(),0))
                dopamine_surge.run(poisson_train(np.ones(1, dtype=np.float), tki.dt(), 0))
        else:
            o, p, m = my_grid.get_state()

            g1_obstacles.run(poisson_train(o, tki.dt(), 74))
            g1_meal.run(poisson_train(m, tki.dt(), 74))
            g1_agent.run(poisson_train(p, tki.dt(), 74))
        
        # run all layers
        nn.run_order(["obstacle_inputs", "meal_inputs", "agent_inputs", "al1", "al1i", "al2", "dopamine_surge", "dopamine_dip", "go", "nogo", "action", "actioni"])
        
        sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

        if step >= duration/tki.dt():
            break

