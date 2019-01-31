from units import *
from timekeeper import TimeKeeperIterator
import random
import numpy as np
from typing import List


class SpikingNeuron:
    inhibitory=0
    excitatory=1
    def __init__(self, neuron_type, group_scope="single"):
        self.n_type = neuron_type
        # adjustable constant parameters
        self.v_rest = -70.0 * mvolt
        self.v_inhibition = -70.0 * mvolt
        self.v_excitatory = 0.0 * mvolt
        self.v_leak = -70.0 * mvolt
        self.v_threshold = -50.0 * mvolt
        self.v_spike = 40.0 * mvolt
        self.v_hyperpolarization = -75.0 * mvolt
        self.membrane_capacitance = 200.0 * pfarad
        self.membrane_time_constant = 2.0 * msec
        self.refractory_period = 2.0 * msec
        self.leak_conductance = 10.0 * nsiem
        self.max_q = 0.004 * ncoul

        # current membrane potential
        self.v_membrane = self.v_rest

        # spikes that have arrived since last evaluation step
        self.dendritic_spikes = []

        # # neurons that this axon projects too
        self.axonal_synapses = []

        # timestamp of last spike
        # self.last_spiked = 0

        # variables to track
        self.tracked_vars = []
        self.charge_track = []
        self.voltage_track = []
        self.spike_track = []

    def reset(self):
        self.v_membrane = self.v_hyperpolarization # self.v_rest
    
    def add_spike(self, s):
        self.dendritic_spikes.append(s)
    
    def q_t(self):
        q = 0.0
        for spike in self.dendritic_spikes:
            if spike['neuron_type'] == self.__class__.inhibitory:
                q += spike['weight'] * (self.max_q * (self.v_inhibition - self.v_membrane) / (self.v_threshold - self.v_inhibition))
            elif spike['neuron_type'] == self.__class__.excitatory:
                q += spike['weight'] * (self.max_q * (self.v_excitatory - self.v_membrane) / (self.v_excitatory - self.v_rest))

        self.dendritic_spikes = []

        return q
    
    def update(self, dt, timeunit):
        output = self.v_membrane
        input_charge = self.q_t()
        
        if True:
            # decay membrane potential
            self.v_membrane -= (self.v_membrane - self.v_rest)*(1.0-np.exp(-dt/self.membrane_time_constant))
                
            # increase membrane potential due to spikes
            self.v_membrane += input_charge / self.membrane_capacitance
            
            output = self.v_membrane
            # check if ready to fire
            if self.v_membrane >= self.v_threshold:
                self.fire()
                output = self.v_spike
        
        # record tracked variables
        if "v_m" in self.tracked_vars:
            self.voltage_track.append(output)
        if "q_t" in self.tracked_vars:
            self.charge_track.append(input_charge)
        if "s_t" in self.tracked_vars:
            if output == self.v_spike:
                self.spike_track.append(1)
            else:
                self.spike_track.append(0)
        
        return output

    def fire(self):
        # self.last_spiked = tstamp
        for synapse in self.axonal_synapses:
            synapse.post_n.dendritic_spikes.append({'neuron_type': self.n_type, 'weight': synapse.w})
        self.reset()


class Synapse:
    def __init__(self, id, pre_n: SpikingNeuron, post_n: SpikingNeuron, w: float=1.0):
        self.pre_neuron = pre_n
        self.post_n = post_n
        self.w = w
        self.id = 1


if __name__ == "__main__":
    tki = TimeKeeperIterator(timeunit=0.1*msec)

    n = SpikingNeuron(SpikingNeuron.excitatory)
    n.refractory_period = 0.0
    n.tracked_vars = ['q_t', 'v_m', 's_t']

    duration = 100 * msec
    input_period = 10.0 * msec

    lts = 0
    for step in tki:
        # if step == 50:
        #     n.add_spike({'neuron_type': SpikingNeuron.excitatory, 'weight': 1.0})
        # generate a spike every millisecond
        if (step - lts)*tki.dt() >= input_period:
            lts = step
            for x in range(np.random.randint(1, 3)):
                r = random.random()
                if r <= 1.0:
                    n.add_spike({'neuron_type': 1, 'weight': 0.8})
        
        n.update(tki.dt(), tki.dt())
        if step == duration/tki.dt():
            break
    
    import matplotlib.pyplot as plt
    f, axarr = plt.subplots(3, sharex=True)
    f.suptitle('Single Neuron')
    axarr[0].plot(n.voltage_track)
    axarr[1].plot(n.current_track)
    axarr[2].plot(n.spike_track, 'r')
    plt.show()
