from timekeeper import TimeKeeperIterator
from NeuralGroup import AdExNeuralGroup, SensoryNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import AdExParams
from units import *
from utils import poisson_train
import numpy as np
import time
import sys
import matplotlib.pyplot as plt


if __name__ == "__main__":
    mode = "single_neuron"
    mode = "single_network"
    # single neuron simulation
    if mode == "single_neuron":
        tki = TimeKeeperIterator(timeunit=0.1*msec)
        duration = 1000.0 * msec
        n = AdExNeuralGroup(np.ones((1,1), dtype=np.int), "George", tki, AdExParams())
        n.tracked_vars = ["v_m", "adap", "i_syn"]

        for step in tki:
            # constant current input
            n.run(-1.0 * namp)[0][0]
            
            sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

            if step >= duration/tki.dt():
                break

        v = [i[0] for i in n.v_m_track] 
        w = [i[0] for i in n.adap_track] 
        isyn = n.isyn_track

        times = np.arange(0,len(v), 1) * tki.dt() / msec

        plt.plot(times, v)
        plt.title("Voltage Track")
        plt.xlabel("Time (msec)")
        plt.ylabel("Membrane Potential (volt)")
        plt.show()

        plt.plot(times, w)
        plt.title("Adaptation Current Track")
        plt.xlabel("Time (msec)")
        plt.ylabel("Adaptation Current (amp)")
        plt.show()

        plt.plot(times, isyn)
        plt.title("Synaptic Current Track")
        plt.xlabel("Time (msec)")
        plt.ylabel("Synaptic Current (amp)")
        plt.show()
    
    if mode == "single_network":
        start = time.time()
        tki = TimeKeeperIterator(timeunit=0.1*msec)
        duration = 1000.0 * msec
        g1 = SensoryNeuralGroup(np.ones(2, dtype=np.int), "Luny", tki, AdExParams())
        g2 = AdExNeuralGroup(np.ones(3, dtype=np.int), "George", tki, AdExParams())
        g3 = AdExNeuralGroup(np.ones(2, dtype=np.int), "Ada", tki, AdExParams())
        g3.tracked_vars = ["v_m", "i_syn", "adap"]

        nn = NeuralNetwork([g1, g2, g3], "blah", tki)
        nn.fully_connect("Luny", "George", w_i=1.0, trainable=True)
        nn.fully_connect("George", "Ada", w_i=1.0, trainable=True)
        
        for step in tki:
            # inject spikes into sensory layer
            g1.run(poisson_train(np.array([0.0, 1.0]), tki.dt(), 64))
            # run all layers
            nn.run_order(["Luny", "George", "Ada"])
            
            sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

            if step >= duration/tki.dt():
                break

        v = [i[0] for i in g3.v_m_track] 
        w = [i[0] for i in g3.adap_track] 
        isyn = [i[0] for i in g3.isyn_track] 

        times = np.arange(0,len(v), 1) * tki.dt() / msec

        plt.plot(times, v)
        plt.title("Voltage Track")
        plt.xlabel("Time (msec)")
        plt.ylabel("Membrane Potential (volt)")
        plt.show()

        plt.plot(times, w)
        plt.title("Adaptation Current Track")
        plt.xlabel("Time (msec)")
        plt.ylabel("Adaptation Current (amp)")
        plt.show()

        plt.plot(times, isyn)
        plt.title("Synaptic Current Track")
        plt.xlabel("Time (msec)")
        plt.ylabel("Synaptic Current (amp)")
        plt.show()

