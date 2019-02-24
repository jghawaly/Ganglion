from units import *
import random
import numpy as np
from typing import List
from learning import dw


class SpikingNeuron:
    inhibitory=0
    excitatory=1
    dci=3
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

        # spikes that have arrived at the current evaluation step
        self.current_dendritic_spikes = []

        # synapses that this axon projects too
        self.axonal_synapses = []

        # synapse that project to this neuron
        self.dendritic_synapses = []

        # variables to track
        self.tracked_vars = []
        self.charge_track = []
        self.voltage_track = []
        self.spike_track = []

        self.group_scope = group_scope

    def reset(self):
        self.v_membrane = self.v_hyperpolarization
    
    def q_t(self):
        q = 0.0
        for spike in self.dendritic_spikes:
            # spike coming from an inhibitory Neuron
            if spike['neuron_type'] == self.__class__.inhibitory:
                q += spike['synapse'].w * (self.max_q * (self.v_inhibition - self.v_membrane) / (self.v_threshold - self.v_inhibition))
            # spike coming from an excitatory Neuron
            elif spike['neuron_type'] == self.__class__.excitatory:
                q += spike['synapse'].w * (self.max_q * (self.v_excitatory - self.v_membrane) / (self.v_excitatory - self.v_rest))
            # spike coming from an artificial direct charge injection
            elif spike['neuron_type'] == self.__class__.dci:
                q += spike['weight'] * (self.max_q * (self.v_excitatory - self.v_membrane) / (self.v_excitatory - self.v_rest))

        self.dendritic_spikes = []

        return q
    
    def update(self):
        self.dendritic_spikes = self.current_dendritic_spikes.copy()
        self.current_dendritic_spikes = []
    
    def evaluate(self, dt, current_timestep):
        output = self.v_membrane
        input_charge = self.q_t()
        
        # decay membrane potential
        self.v_membrane -= (self.v_membrane - self.v_rest)*(1.0-np.exp(-dt/self.membrane_time_constant))
            
        # increase membrane potential due to spikes
        self.v_membrane += input_charge / self.membrane_capacitance
        
        output = self.v_membrane
        # check if ready to fire
        if self.v_membrane >= self.v_threshold:
            self.fire(current_timestep)
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

    def fire(self, current_timestep):
        # self.last_spiked = tstamp
        for synapse in self.axonal_synapses:
            synapse.post_n.current_dendritic_spikes.append({'neuron_type': self.n_type, 'synapse': synapse, 'timestep': current_timestep})
            synapse.pre_spikes.append(current_timestep)
        
        for synapse in self.dendritic_synapses:
            synapse.post_spikes.append(current_timestep)

        self.reset()


class Synapse:
    def __init__(self, id, pre_n: SpikingNeuron, post_n: SpikingNeuron, w: float=1.0):
        self.pre_neuron = pre_n
        self.post_n = post_n
        self.w = w
        self.id = 1

        self.window = 50.0 * msec
        
        self.pre_spikes = []
        self.post_spikes = []
    
    def stdp(self):
        # NOTE: Need to watch this carefully, could become a potential memory leak if the pre and postsynaptic spike history is not dumped
        if len(self.post_spikes) == 2:
            # first postsynaptic spike time
            t1 = self.post_spikes[0]
            # second postsynaptic spike time
            t2 = self.post_spikes[1]

            # parse out presynaptic spikes that have an effect on the spike being evaluated
            relevant_pre = [t for t in self.pre_spikes if t < t2]

            # delta time between the postsynaptic spike being evaluated and all relevant presynaptic spikes
            dt = [t1 - t for t in relevant_pre]

            # calculate total synaptic weight change (should this be recursive?)
            dw_total = np.array([dw(val, 0.5, 10 * msec, 10 * msec, 1.0, 1.0) for val in dt]).sum()
            
            self.w += dw_total
            print("weight_changed: %g" % dw_total)

            # remove presynaptic spikes that occured before the spike that STDP was just evaluated on
            self.pre_spikes = [t for t in self.pre_spikes if t > t1]

            # remove the post synaptic spike that was just evaluated
            self.post_spikes = [self.post_spikes[1]]
        elif len(self.post_spikes) == 1:
            # cull presynaptic spikes that are outside of our window. We need to keep spikes that come 2*window after the postsynaptic
            # spike, otherwise we may cull presynaptic spikes to the NEXT postsynaptic spike, which has not yet arrived
            self.pre_spikes = [t for t in self.pre_spikes if -2*self.window <= self.post_spikes[0] - t <= self.window]
        elif len(self.post_spikes) == 0:
            pass
        else:
            print("More than 2 postsynaptic spikes encountered in synaptic history, that should not happen.")



        