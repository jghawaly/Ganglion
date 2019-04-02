from timekeeper import TimeKeeperIterator
from Neuron import AdExNeuron, AdExParams
from units import *
import matplotlib.pyplot as plt
import numpy as np
import sys

# Single neuron example

if __name__ == "__main__":
    # single spike injection

    tki = TimeKeeperIterator(timeunit=0.1*msec)
    duration = 100.0 * msec

    n = AdExNeuron(AdExNeuron.excitatory, AdExParams(), tki)
    n.tracked_vars = ["v_m", "s_t", "wadex"]

    for step in tki:
        # inject a spike at 50 msec
        if tki.tick_time() == 25 * msec:
            n.add_spike({'neuron_type': AdExNeuron.desi, 'weight': 1.0, 'timestep': tki.tick_time()})
        # n.add_spike({'neuron_type': AdExNeuron.dci, 'weight': -1 * namp, 'timestep': tki.tick_time()})
            
        n.evaluate()

        sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

        if step >= duration/tki.dt():
            break

    times = np.arange(0,len(n.voltage_track), 1) * tki.dt() / msec
    plt.plot(times, n.voltage_track)
    plt.title("Voltage Track")
    plt.xlabel("Time (msec)")
    plt.ylabel("Membrane Potential (mvolt)")
    plt.show()

    plt.plot(times, n.spike_track)
    plt.title("Spike Track")
    plt.xlabel("Time (msec)")
    plt.ylabel("Spike Events (a.u.)")
    plt.show()

    plt.plot(times, n.wadex_track)
    plt.title("Adaptation Conductance Track")
    plt.xlabel("Time (msec)")
    plt.ylabel("Adaptation Conductance (nsiem)")
    plt.show()
    
