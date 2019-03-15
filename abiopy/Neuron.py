from units import *
import random
import numpy as np
from typing import List, Dict
from learning import dw
from numba import jit


class NeuronParams:
    def __init__(self):
        self.v_rest = -70.0 * mvolt
        self.v_inhibition = -75.0 * mvolt
        self.v_excitatory = 0.0 * mvolt
        self.v_leak = -70.0 * mvolt
        self.v_threshold = -50.0 * mvolt
        self.v_spike = 40.0 * mvolt
        self.v_hyperpolarization = -80.0 * mvolt
        self.membrane_capacitance = 200.0 * pfarad 
        self.membrane_time_constant = 10.0 * msec
        self.leak_conductance = 10.0 * nsiem
        self.max_q = 1.0 * pcoul # 4 pcoul should be enough to trigger with default settings


@jit(nopython=True)
def calc_q(weight: float, max_q: float, v_mem: float, v_diff: float, v_top: float, v_bot: float):
    return weight * (max_q * (v_diff - v_mem) / (v_top - v_bot))


class SpikingNeuron:
    inhibitory=0
    excitatory=1
    dci=3
    def __init__(self, neuron_type, params: NeuronParams, group_scope="single"):
        self.n_type = neuron_type
        # adjustable constant parameters
        self.v_rest = params.v_rest
        self.v_inhibition = params.v_inhibition
        self.v_excitatory = params.v_excitatory
        self.v_leak = params.v_leak
        self.v_threshold = params.v_threshold
        self.v_spike = params.v_spike
        self.v_hyperpolarization = params.v_hyperpolarization
        self.membrane_capacitance = params.membrane_capacitance
        self.membrane_time_constant = params.membrane_time_constant
        self.leak_conductance = params.leak_conductance
        self.max_q = params.max_q

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

        # decay membrane potential
        self.v_membrane = self.v_membrane - (self.v_membrane - self.v_rest) * (1.0 - np.exp(-dt / self.membrane_time_constant))
        
        q_total = 0.0
        # disable membrane potentiation for however long it takes for the neuron's membrane potential to be released from the
        # post-spike hyperpolarization
        if self.v_membrane >= self.v_inhibition:
            for spike in self.dendritic_spikes:
                q = 0.0
                # spike coming from an inhibitory Neuron
                if spike['neuron_type'] == self.__class__.inhibitory:
                    # q = spike['synapse'].w * (self.max_q * (self.v_inhibition - self.v_membrane) / (self.v_threshold - self.v_inhibition))
                    q = calc_q(spike['synapse'].w, self.max_q, self.v_membrane, self.v_inhibition, self.v_threshold, self.v_inhibition)
                # spike coming from an excitatory Neuron
                elif spike['neuron_type'] == self.__class__.excitatory:
                    # q = spike['synapse'].w * (self.max_q * (self.v_excitatory - self.v_membrane) / (self.v_excitatory - self.v_rest))
                    q = calc_q(spike['synapse'].w, self.max_q, self.v_membrane, self.v_excitatory, self.v_excitatory, self.v_rest)
                # spike coming from an artificial direct charge injection
                elif spike['neuron_type'] == self.__class__.dci:
                    # q = spike['weight'] * (self.max_q * (self.v_excitatory - self.v_membrane) / (self.v_excitatory - self.v_rest))
                    q = calc_q(spike['weight'], self.max_q, self.v_membrane, self.v_excitatory, self.v_excitatory, self.v_rest)
                
                # update membrane potential
                self.v_membrane = self.v_membrane + q / self.membrane_capacitance
                # for tracking charge injection
                q_total += q

        self.dendritic_spikes = []

        # this may occur when inhibitory weights greater than 1.0 occur, useful for forcing a neuron to spike by raising weight to
        # a very large number
        if self.v_membrane < self.v_hyperpolarization:
            self.v_membrane = self.v_inhibition

        output = self.v_membrane
        # check if ready to fire
        if self.v_membrane >= self.v_threshold:
            self.fire(current_timestep)
            output = self.v_spike
        
        # record tracked variables
        if "v_m" in self.tracked_vars:
            self.voltage_track.append(output)
        if "q_t" in self.tracked_vars:
            self.charge_track.append(q_total)
        if "s_t" in self.tracked_vars:
            if output == self.v_spike:
                self.spike_track.append(1)
            else:
                self.spike_track.append(0)
        
        return output
    
    def evaluate2(self, dt, current_timestep):
        output = self.v_membrane
        input_charge = self.q_t()
        
        # update membrane potential
        self.v_membrane = self.v_membrane + input_charge / self.membrane_capacitance - (self.v_membrane - self.v_rest) * (1.0 - np.exp(-dt / self.membrane_time_constant))
        
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
            # synapse.post_n.current_dendritic_spikes.append({'neuron_type': self.n_type, 'synapse': synapse, 'timestep': current_timestep})
            synapse.post_n.dendritic_spikes.append({'neuron_type': self.n_type, 'synapse': synapse, 'timestep': current_timestep}) 
            synapse.pre_spikes.append(current_timestep)
        
        for synapse in self.dendritic_synapses:
            synapse.post_spikes.append(current_timestep)

        self.reset()
