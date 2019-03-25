from timekeeper import TimeKeeperIterator
from Neuron import AdExNeuron, AdExParams
from units import *
import matplotlib.pyplot as plt
import numpy as np

# Single neuron example

if __name__ == "__main__":
    n = AdExNeuron(AdExNeuron.excitatory, AdExParams())
    n.tracked_vars = ["v_m", "s_t", "wadex"]

    tki = TimeKeeperIterator(timeunit=0.1*msec)
    duration = 300.0 * msec
    for step in tki:
        # inject a spike at 50 msec
        if 150*msec < tki.tick_time() <= 200*msec:
            n.add_spike({'neuron_type': AdExNeuron.disi, 'weight': 0.1, 'timestep': tki.tick_time()})
             
        n.evaluate(tki.dt(), tki.tick_time())

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
    