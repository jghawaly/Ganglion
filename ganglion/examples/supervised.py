import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, ExLIFNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import ExLIFParams
from units import *
from utils import poisson_train
import numpy as np
import time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    inputs = (np.array([[1, 1, 1],[1, 0, 0],[1, 0, 0]]), np.array([[0, 0, 0],[0, 0, 1],[1, 1, 1]]), np.array([[0, 1, 0],[1, 1, 1],[0, 1, 0]]))
    desired = (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    start = time.time()
    tki = TimeKeeperIterator(timeunit=0.1*msec)
    duration = 5000.0 * msec
    g1 = SensoryNeuralGroup(np.ones(9, dtype=np.int), "INPUTS", tki, ExLIFParams(), field_shape=(3,3))
    g2 = ExLIFNeuralGroup(np.ones(18, dtype=np.int), "HIDDEN", tki, ExLIFParams())
    g2i = ExLIFNeuralGroup(np.zeros(18, dtype=np.int), "HIDDEN_I", tki, ExLIFParams())
    g3 = ExLIFNeuralGroup(np.ones(3, dtype=np.int), "OUTPUTS", tki, ExLIFParams())
    g3i = ExLIFNeuralGroup(np.zeros(3, dtype=np.int), "OUTPUTS_I", tki, ExLIFParams())
    g4 = SensoryNeuralGroup(np.ones(3, dtype=np.int), "TEACHER", tki, ExLIFParams())
    g3.tracked_vars = ["spike", "v_m"]

    nn = NeuralNetwork([g1, g2, g2i, g3, g3i, g4], "supervised", tki)
    nn.fully_connect("INPUTS", "HIDDEN", trainable=True, minw=0.01, maxw=0.2)
    nn.one_to_one_connect("HIDDEN", "HIDDEN_I", trainable=False, w_i=1.0)
    nn.fully_connect("HIDDEN_I", "HIDDEN", trainable=False, w_i=1.0, skip_one_to_one=True)
    nn.fully_connect("HIDDEN", "OUTPUTS", trainable=True, minw=0.01, maxw=0.1)
    nn.one_to_one_connect("OUTPUTS", "OUTPUTS_I", trainable=False, w_i=1.0)
    nn.fully_connect("OUTPUTS_I", "OUTPUTS", trainable=False, w_i=1.0, skip_one_to_one=True)
    nn.one_to_one_connect("TEACHER", "OUTPUTS", trainable=False, w_i=1.0)
    
    input_index = 0
    current_input = inputs[input_index]
    current_teacher = desired[input_index]

    lts = 0
    # skip = False
    for step in tki:
        if (step - lts)*tki.dt() >= 100*msec:
            input_index += 1
            if input_index == 3:
                input_index = 0
            lts = step
            current_input = inputs[input_index]
            current_teacher = desired[input_index]

            spikes = np.array(g3.spike_track)
            rate = np.sum(spikes, axis=0) * 100.0 /np.sum(spikes) if np.sum(spikes) >0 else np.sum(spikes, axis=0)
            print()
            print(current_teacher)
            print(rate)
            print()

            # plt.plot(g3.v_m_track)
            # plt.show()
            # skip = not skip
        
        # inject spikes into sensory and teaching layer
        g1.run(poisson_train(current_input, tki.dt(), 64))
        g4.run(poisson_train(current_teacher, tki.dt(), 500))

        # run all layers
        nn.run_order(["INPUTS", "HIDDEN" ,"HIDDEN_I", "TEACHER", "OUTPUTS", "OUTPUTS_I"])
        
        sys.stdout.write("Current simulation time: %g milliseconds       \r" % (step * tki.dt() / msec))

        if step >= duration/tki.dt():
            break

    