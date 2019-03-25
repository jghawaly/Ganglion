import sys
from timekeeper import TimeKeeperIterator
from NeuralGroup import NeuralGroup, StructuredNeuralGroup
from NeuralNetwork import NeuralNetwork
from Neuron import SpikingNeuron, NeuronParams
from units import *
import random
import matplotlib.pyplot as plt
import sys
import numpy as np

# Single neuron example

if __name__ == "__main__":
    tki = TimeKeeperIterator(timeunit=0.01*msec)
    my_np = NeuronParams()

    g1 = NeuralGroup(0, 1, "neuron", neuron_params=my_np)
    g1.track_vars(['q_t', 'v_m', 's_t'])

    nn = NeuralNetwork([g1], "my_net")

    duration = 30 * msec
    input_period = 0.5 * msec

    lts = 0
    for step in tki:
        if (step - lts)*tki.dt() >= input_period:
            lts = step
            for x in range(1):
                g1.dci(np.ones(g1.shape), tki.tick_time())
             
        nn.run_order(["neuron"], tki)

        if step >= duration/tki.dt():
            break
    
    n = g1.n[0]

    plt.plot(n.voltage_track)
    plt.show()