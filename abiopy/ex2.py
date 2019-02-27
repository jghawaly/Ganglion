import sys
from timekeeper import TimeKeeperIterator
from NeuralGroup import NeuralGroup
from NeuralNetwork import NeuralNetwork
from Neuron import SpikingNeuron
from units import *
import random
import numpy as np


tki = TimeKeeperIterator(timeunit=0.01*msec)

g1 = NeuralGroup(0, 16, "input")
g2 = NeuralGroup(2, 2, "hidden")
g3 = NeuralGroup(0, 2, "output")
g4 = NeuralGroup(2, 0, "output_inhib")

nn = NeuralNetwork([g1, g2, g3, g4], "my_net")
nn.fully_connect(0, 1)
nn.fully_connect(1, 2)
nn.fully_connect(1, 3)
nn.fully_connect(3, 2)

g3.track_vars(['q_t', 'v_m', 's_t'])
g1.track_vars(['q_t', 'v_m', 's_t'])
g2.track_vars(['q_t', 'v_m', 's_t'])

duration = 1000 * msec
input_period = 1.0 * msec

volts = []

# i1 = np.zeros(30)
# i1[0:15]= 1.0
# i2 = np.zeros(30)
# i2[15:] = 1.0
i1 = np.array([1.0, 1.0, 1.0, 1.0,
               0.0, 1.0, 1.0, 1.0,
               0.0, 0.0, 1.0, 1.0,
               0.0, 0.0, 0.0, 1.0])
i2 = np.array([0.0, 1.0])

lts = 0
tick = False
for step in tki:
    if (step - lts)*tki.dt() >= input_period:
        lts = step
        if tick:
            g1.dci(i1)
            g1.dci(i1)
            if step < 5000:
                g3.dci(i2)
                g3.dci(i2)

        tick = not tick
            
    nn.run_order([0, 1, 3, 2], tki)

    volts.append(g2.spike_track)

    if step == duration/tki.dt():
        break

import matplotlib.pyplot as plt

volts = np.array(volts)
volts = volts + volts.min()

plt.plot(volts[:,0])
plt.plot(volts[:,1] + 0.2)
plt.show()
