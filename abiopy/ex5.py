from utils import load_mnist
from timekeeper import TimeKeeperIterator
from NeuralGroup import NeuralGroup, StructuredNeuralGroup, weight_map_between
from NeuralNetwork import NeuralNetwork
from Neuron import NeuronParams, SpikingNeuron
from units import *
import random
import numpy as np
import sys
import os
import matplotlib.pyplot as plt


# load MNIST
train_data, train_labels, test_data, test_labels = load_mnist()

params = NeuronParams()
params.membrane_time_constant = 5.0 * msec

# set up SNN time tracker
tki = TimeKeeperIterator(timeunit=0.5*msec)
duration = 5000 * msec
input_period = 1.0 * msec

lgn = StructuredNeuralGroup(np.ones((28, 28)), 'lgn', neuron_params=params)
v1_exc = StructuredNeuralGroup(np.ones((10, 10)), 'v1_exc', neuron_params=params)
v1_inh = StructuredNeuralGroup(np.zeros((10, 10)), 'v1_inh', neuron_params=params)

v1_inh.track_vars(['q_t', 'v_m', 's_t'])

nn = NeuralNetwork([lgn, v1_exc, v1_inh], "NN")
nn.fully_connect("lgn", "v1_exc")
nn.one_to_one("v1_exc", "v1_inh", w_i=1.9, trainable=False)
nn.fully_connect("v1_inh", "v1_exc", skip_self=True, w_i=1.9, trainable=False)

# clear stdout
os.system('cls' if os.name == 'nt' else 'clear')

lts = 0
lts2 = 0
mnist_counter = 0
for step in tki:
    if (step - lts2)*tki.dt() >= 100*input_period:
        mnist_counter += 1
    if (step - lts)*tki.dt() >= input_period:
        lts = step
        lgn.dci(train_data[mnist_counter])# * input_period / (1.0*msec)) # 17
            
    nn.run_order(["lgn", "v1_exc", "v1_inh"], tki, lr_ex=0.01, lr_inh=0.01)

    if step >= duration/tki.dt():
        break
    sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))
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
plt.show()
