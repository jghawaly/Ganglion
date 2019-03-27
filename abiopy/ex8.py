from utils import load_mnist, poisson_train
from timekeeper import TimeKeeperIterator
from NeuralGroup import NeuralGroup, StructuredNeuralGroup, weight_map_between, vmem_map, spike_map
from NeuralNetwork import NeuralNetwork
from Neuron import AdExParams, AdExNeuron
from Synapse import STDPParams
from units import *
import random
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *


params = AdExParams()
params.refractory_period = 0
l_params = STDPParams()
l_params.tao_plus = 3 * msec
l_params.tao_minus = 3 * msec
l_params.lr_minus = 0.001
l_params.lr_plus = 0.001
l_params.a_plus = 1.0
l_params.a_minus = -1.0

# set up SNN time tracker
tki = TimeKeeperIterator(timeunit=0.1*msec)
duration = 500 * msec

lgn = StructuredNeuralGroup(np.ones((8, 8)), 'lgn', tki, neuron_params=params)
v1_exc = StructuredNeuralGroup(np.ones((4, 8)), 'v1_exc', tki, neuron_params=params)
v1_inh = StructuredNeuralGroup(np.zeros((4, 8)), 'v1_inh', tki, neuron_params=params)

# v1_inh.track_vars(['q_t', 'v_m', 's_t'])
v1_exc.track_vars(['v_m', 's_t'])

nn = NeuralNetwork([lgn, v1_exc, v1_inh], "NN", tki)
nn.fully_connect("lgn", "v1_exc", learning_params=l_params, minw=0.1, maxw=0.3)
nn.one_to_one("v1_exc", "v1_inh", w_i=0.8, trainable=False)
nn.fully_connect("v1_inh", "v1_exc", skip_self=True, w_i=0.8, trainable=False)

def genbar(s1, s2):
    i = np.zeros((s1, s2), dtype=np.float)
    if random.random() > 0.5:
        i[:, np.random.randint(0, s2)] = 1.0
    else:
        i[np.random.randint(0, s1), :] = 1.0
    
    return i
    
def genbars(s1, s2):
    i = np.zeros((s1, s2), dtype=np.float)
    for _ in range(2):
        i[:, np.random.randint(0, s2)] = 1.0
    
    for _ in range(2):
        i[np.random.randint(0, s1), :] = 1.0
    
    return i

arr = []
for neuron in v1_exc.n:
    wmap = weight_map_between(lgn, neuron)
    arr.append(wmap)
arr = np.array(arr)
nr = v1_exc.shape[0]
nc = v1_exc.shape[1]
for row in range(nr):
    row_img = np.hstack(arr[row*nc:row*nc +nc, :, :])
    if row == 0:
        img = row_img
    else:
        img = np.vstack((img, row_img))

plt.imshow(img)
plt.colorbar()
plt.clim(0, 1)
plt.show()

d = genbar(8, 8)
lts = 0
for step in tki:
    if (step - lts)*tki.dt() >= 50*msec:
        nn.rest()
        lts = step
        d = genbar(8, 8)
    
    lgn.direct_injection(poisson_train(d, tki.dt(), 64), tki.tick_time(), AdExNeuron.desi)
    nn.run_order(["lgn", "v1_exc", "v1_inh"], tki)

    sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))
    # if (step - lts)*tki.dt() <= 3*msec:
    #     plt.imshow(spike_map(v1_exc))
    #     # plt.clim(-80.0 * mvolt, 40 * mvolt)
    #     plt.clim(0, 1)
    #     plt.colorbar()
    #     plt.show()
        
    if step >= duration/tki.dt():
        break
    
print("\n\n")
arr = []
for neuron in v1_exc.n:
    wmap = weight_map_between(lgn, neuron)
    arr.append(wmap)
arr = np.array(arr)
nr = v1_exc.shape[0]
nc = v1_exc.shape[1]
for row in range(nr):
    row_img = np.hstack(arr[row*nc:row*nc +nc, :, :])
    if row == 0:
        img = row_img
    else:
        img = np.vstack((img, row_img))

plt.imshow(img)
plt.colorbar()
plt.clim(0, 1)
plt.show()
