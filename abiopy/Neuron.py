from units import *
import random
import numpy as np
from typing import List, Dict
from learning import dw
from numba import jit




# @jit(nopython=True)
# def calc_q(weight: float, max_q: float, v_mem: float, v_diff: float, v_top: float, v_bot: float):
#     return weight * (max_q * (v_diff - v_mem) / (v_top - v_bot))


# class SpikingNeuron:
#     inhibitory=0
#     excitatory=1
#     dci=2
#     spiker=3
#     def __init__(self, neuron_type, params: NeuronParams, group_scope="single"):
#         self.n_type = neuron_type
#         # adjustable constant parameters
#         self.v_rest = params.v_rest
#         self.v_inhibition = params.v_inhibition
#         self.v_excitatory = params.v_excitatory
#         self.v_leak = params.v_leak
#         self.v_spike = params.v_spike
#         self.v_hyperpolarization = params.v_hyperpolarization
#         self.v_threshold_min = params.v_threshold_min
#         self.membrane_capacitance = params.membrane_capacitance
#         self.membrane_time_constant = params.membrane_time_constant
#         self.threshold_time_constant = params.threshold_time_constant
#         self.leak_conductance = params.leak_conductance
#         self.max_q = params.max_q
#         self.ar_enabled = params.absolute_refractoriness_enabled

#         # current membrane potential
#         self.v_membrane = self.v_rest

#         # initial spike threshold
#         # self.v_threshold_initial = params.v_threshold

#         # dynamic spike threshold set at initial value
#         self.v_threshold = params.v_threshold

#         # spikes that have arrived since last evaluation step
#         self.dendritic_spikes = []

#         # synapses that this axon projects too
#         self.axonal_synapses = []

#         # synapse that project to this neuron
#         self.dendritic_synapses = []

#         # variables to track
#         self.tracked_vars = []
#         self.charge_track = []
#         self.voltage_track = []
#         self.spike_track = []

#         # indicates if this neuron spiked during the last evaluation step
#         self.spiked = 0

#         # indicates the time at which the neuron last spiked
#         self.last_spike_time = 0.0

#         self.group_scope = group_scope

#     def reset(self):
#         self.v_membrane = self.v_hyperpolarization
    
#     def evaluate(self, dt, current_timestep, absolute_refractoriness=True):
#         output = self.v_membrane
#         self.spiked = 0

#         # decay membrane potential
#         self.v_membrane = self.v_membrane - (self.v_membrane - self.v_rest) * (1.0 - np.exp(-dt / self.membrane_time_constant))

#         # decay spike threshold
#         self.v_threshold = self.v_threshold - (self.v_threshold - self.v_threshold_min) * (1.0 - np.exp(-dt / self.threshold_time_constant))
        
#         # used for tracking the input charge
#         q_total = 0.0

#         # disable membrane potentiation for however long it takes for the neuron's membrane potential to be released from the
#         # post-spike hyperpolarization IF this feature is enabled for this neuron
#         if self.ar_enabled:
#             if self.v_membrane >= self.v_inhibition:
#                 in_refractory_period = False
#             else:
#                 in_refractory_period = True
#         else:
#             in_refractory_period = False
        
#         if not in_refractory_period:
#             for spike in self.dendritic_spikes:
#                 q = 0.0
#                 # spike coming from an inhibitory Neuron
#                 if spike['neuron_type'] == self.__class__.inhibitory:
#                     # q = spike['synapse'].w * (self.max_q * (self.v_inhibition - self.v_membrane) / (self.v_threshold - self.v_inhibition))
#                     q = calc_q(spike['synapse'].w, self.max_q, self.v_membrane, self.v_inhibition, self.v_threshold, self.v_inhibition)
#                 # spike coming from an excitatory Neuron
#                 elif spike['neuron_type'] == self.__class__.excitatory:
#                     # q = spike['synapse'].w * (self.max_q * (self.v_excitatory - self.v_membrane) / (self.v_excitatory - self.v_rest))
#                     q = calc_q(spike['synapse'].w, self.max_q, self.v_membrane, self.v_excitatory, self.v_excitatory, self.v_rest)
#                 # spike coming from an artificial direct charge injection
#                 elif spike['neuron_type'] == self.__class__.dci:
#                     # q = spike['weight'] * (self.max_q * (self.v_excitatory - self.v_membrane) / (self.v_excitatory - self.v_rest))
#                     q = calc_q(spike['weight'], self.max_q, self.v_membrane, self.v_excitatory, self.v_excitatory, self.v_rest)
#                 elif spike['neuron_type'] == self.__class__.spiker:
#                     # calculate the charge needed to cause the neuron to spike. We could do this in a more forceful way by simply calling fire(),
#                     # but this method is more compatible with the current way that this method works
#                     q = self.membrane_capacitance * (self.v_threshold - self.v_membrane)
                
#                 # update membrane potential
#                 self.v_membrane = self.v_membrane + q / self.membrane_capacitance
#                 # for tracking charge injection
#                 q_total += q

#         self.dendritic_spikes = []

#         # this may occur when inhibitory weights greater than 1.0 occur, useful for forcing a neuron to spike by raising weight to
#         # a very large number
#         if self.v_membrane < self.v_hyperpolarization:
#             self.v_membrane = self.v_inhibition

#         output = self.v_membrane
#         # check if ready to fire
#         if self.v_membrane >= self.v_threshold:
#             self.fire(current_timestep)
#             output = self.v_spike
        
#         # record tracked variables
#         if "v_m" in self.tracked_vars:
#             self.voltage_track.append(output)
#         if "q_t" in self.tracked_vars:
#             self.charge_track.append(q_total)
#         if "s_t" in self.tracked_vars:
#             if self.spiked:
#                 self.spike_track.append(1)
#             else:
#                 self.spike_track.append(0)
        
#         return output

#     def fire(self, current_timestep):
#         """
#         Propagate spike to axonal synapses and reset the current neuron's membrane potential
#         """
#         # update some tracked parameters
#         self.last_spike_time = current_timestep
#         self.spiked = 1
        
#         for synapse in self.axonal_synapses:
#             # propagate the spike across the outgoing axonal synapse
#             synapse.post_n.dendritic_spikes.append({'neuron_type': self.n_type, 'synapse': synapse, 'timestep': current_timestep}) 
#             # notify the axonal synapse that its presynaptic neuron fired at this time step (for Hebbian learning)
#             synapse.pre_spikes.append(current_timestep)
        
#         for synapse in self.dendritic_synapses:
#             # notify all dendritic synapses that the postsynaptic neuron fired at this time step (for Hebbian learning)
#             synapse.post_spikes.append(current_timestep)

#         # increase firing threshold
#         self.v_threshold = self.v_threshold + 0.1 * (self.v_excitatory - self.v_threshold)

#         # reset the neuron's membrane potential to its hyperpolarized value
#         self.reset()


class AdExParams:
    def __init__(self):
        # Parameters from Brette and Gerstner (2005).
        self.v_r = -70.6 * mvolt
        self.v_m = -70.6 * mvolt
        self.v_spike = 40.0 * mvolt
        self.w = 0.0 * namp  
        self.v_thr = -50.4 * mvolt
        self.sf = 2.0 * mvolt
        self.tao_m = 9.37 * msec
        self.c_m = 281.0 * pfarad
        self.a = 4.0 * nsiem
        self.b = 0.0805 * namp
        self.tao_w = 144.0 * msec
        self.i_offset = 0.0 * namp

        # synaptic current parameters
        self.gbar_e = 100.0 * nsiem
        self.gbar_i = 100.0 * nsiem
        self.tao_syn = 5.0 * msec
        self.vrev_e = 0.0 * mvolt
        self.vrev_i = -75.0 * mvolt
        self.spike_window = 20.0 * msec


class AdExNeuron:
    inhibitory=0  # inhibitory neuron
    excitatory=1  # excitatory neuron
    # spike_forcer=2  # forces a spike in the neurons that it projects too
    desi=3  # direct excitatory spike injection
    disi=4  # direct inhibitory spike injection
    dci=5  # direct current injection
    def __init__(self, neuron_type, params: AdExParams, group_scope="single"):
        self.n_type = neuron_type

        # Parameters from Brette and Gerstner (2005).
        self.v_r = params.v_r
        self.v_m = params.v_m
        self.v_spike = params.v_spike
        self.w = params.w
        self.v_thr = params.v_thr
        self.sf = params.sf
        self.tao_m = params.tao_m
        self.c_m = params.c_m
        self.a = params.a
        self.b = params.b
        self.tao_w = params.tao_w
        self.i_offset = params.i_offset

        # synaptic current parameters
        self.gbar_e = params.gbar_e
        self.gbar_i = params.gbar_i
        self.tao_syn = params.tao_syn
        self.vrev_e = params.vrev_e
        self.vrev_i = params.vrev_i
        self.spike_window = params.spike_window

        # spikes that have arrived since last evaluation step
        self.dendritic_spike_times = []
        self.dendritic_spike_weights = []
        self.dendritic_spike_types = []

        # synapses that this axon projects too
        self.axonal_synapses = []

        # synapse that project to this neuron
        self.dendritic_synapses = []

        # variables to track
        self.tracked_vars = []
        self.charge_track = []
        self.voltage_track = []
        self.spike_track = []
        self.wadex_track = []

        # indicates if this neuron spiked during the last evaluation step
        self.spiked = 0

        # indicates the time at which the neuron last spiked
        self.last_spike_time = 0.0

        self.group_scope = group_scope

    def reset(self):
        """
        Reset the membrane potential and adaptation conductance
        """
        self.v_m = self.v_r
        self.w += self.b
    
    def filter_spikes(self, current_timestep):
        """
        Only keep dendritic spikes that are withing the spike evaluation window
        """
        spike_times= np.array(self.dendritic_spike_times)
        spike_weights = np.array(self.dendritic_spike_weights)
        spike_types = np.array(self.dendritic_spike_types)

        # calculate time-since-spike for all dendritic spikes
        tss = current_timestep - spike_times

        windowed_indices = np.where(tss <= self.spike_window)

        self.dendritic_spike_times = spike_times[windowed_indices].tolist()
        self.dendritic_spike_weights = spike_weights[windowed_indices].tolist()
        self.dendritic_spike_types = spike_types[windowed_indices].tolist()
    
    def psp(self, current_timestep):
        """
        Calculate the total input current coming in on the dendritic synapses
        """
        # we only want to keep and evaluate dendritic spikes within the window of interest
        self.filter_spikes(current_timestep)

        # track the total synaptic input current
        i_total = 0.0
        for i in range(len(self.dendritic_spike_times)):
            spike_time = self.dendritic_spike_times[i]
            spike_weight = self.dendritic_spike_weights[i]
            spike_type = self.dendritic_spike_types[i]

            # calculate time since this spike occured
            delta_t = current_timestep - spike_time

            if spike_type == self.__class__.excitatory:
                vrev = self.vrev_e 
                gbar = self.gbar_e
                isyn = spike_weight * (self.v_m - vrev) * gbar * np.exp(-delta_t / self.tao_syn) * np.heaviside(delta_t, 0.0)
            elif spike_type == self.__class__.inhibitory:
                vrev = self.vrev_i
                gbar = self.gbar_i
                isyn = spike_weight * (self.v_m - vrev) * gbar * np.exp(-delta_t / self.tao_syn) * np.heaviside(delta_t, 0.0)
            elif spike_type == self.__class__.desi:
                vrev = self.vrev_e
                gbar = self.gbar_e
                isyn = spike_weight * (self.v_m - vrev) * gbar * np.exp(-delta_t / self.tao_syn) * np.heaviside(delta_t, 0.0)
            elif spike_type == self.__class__.disi:
                vrev = self.vrev_i
                gbar = self.gbar_i
                isyn = spike_weight * (self.v_m - vrev) * gbar * np.exp(-delta_t / self.tao_syn) * np.heaviside(delta_t, 0.0)
            elif spike_type == self.__class__.dci:
                # NOTE: This is a hacky way to make this part work for direct charge injection, the spike "weight" in this case is actually
                # the injection charge, not the weight
                isyn = spike_weight
            # elif spike_type == self.__class__.spike_forcer:
            #     # NOTE: This is also a hacky way to force a spike
            #     isyn = 999999
            else:
                # this should not occur
                raise ValueError("This spike type is invalid: %g" % spike_type)
            
            i_total += isyn
            
        return i_total

    def evaluate(self, dt, current_timestep, absolute_refractoriness=True):
        """
        Update the state of the neuron.
        """
        self.spiked = 0

        # get input current
        self.i_input = self.psp(current_timestep)

        # update membrane potential
        dv_m = dt * (((self.v_r - self.v_m) + self.sf *np.exp((self.v_m - self.v_thr) / self.sf)) / self.tao_m - (self.i_offset + self.i_input + self.w) / self.c_m)
        self.v_m += dv_m

        # update adaptation parameter
        dw = dt * ((self.a * (self.v_m - self.v_r) - self.w) / self.tao_w)
        self.w += dw
        
        # used for tracking the input charge
        q_total = 0.0

        output = self.v_m
        
        # check if ready to fire
        if self.v_m >= self.v_thr:
            self.fire(current_timestep)
            output = self.v_spike
        
        # record tracked variables
        if "v_m" in self.tracked_vars:
            self.voltage_track.append(output)
        if "q_t" in self.tracked_vars:
            self.charge_track.append(q_total)
        if "s_t" in self.tracked_vars:
            if self.spiked:
                self.spike_track.append(1)
            else:
                self.spike_track.append(0)
        if "wadex" in self.tracked_vars:
            self.wadex_track.append(self.w)
        
        return output

    def add_spike(self, spike_descriptor):
        """
        Add a dendritic spike to this neuron
        """
        self.dendritic_spike_times.append(spike_descriptor['timestep']) 
        self.dendritic_spike_weights.append(spike_descriptor['weight']) 
        self.dendritic_spike_types.append(spike_descriptor['neuron_type']) 

    def fire(self, current_timestep):
        """
        Propagate spike to axonal synapses and reset the current neuron to its post-spike state
        """
        # update some tracked parameters
        self.last_spike_time = current_timestep
        self.spiked = 1
        
        for synapse in self.axonal_synapses:
            # propagate the spike across the outgoing axonal synapse
            synapse.post_n.add_spike({'neuron_type': self.n_type, 'weight': synapse.w, 'timestep': current_timestep}) 
            # notify the axonal synapse that its presynaptic neuron fired at this time step (for Hebbian learning)
            synapse.pre_spikes.append(current_timestep)
        
        for synapse in self.dendritic_synapses:
            # notify all dendritic synapses that the postsynaptic neuron fired at this time step (for Hebbian learning)
            synapse.post_spikes.append(current_timestep)

        # reset the neuron's membrane potential to its hyperpolarized value
        self.reset()


if __name__ == "__main__":
    import matplotlib.pyplot as plt 

    n = AdExNeuron(AdExNeuron.excitatory, AdExParams())
    n.tracked_vars = ["v_m", "s_t", "wadex"]
    c = 0
    for val in np.arange(0, 300*msec, 0.1*msec):
        if c == 0:
            n.add_spike({'neuron_type': AdExNeuron.desi, 'weight': 1.0, 'timestep': val})
        n.evaluate(0.1*msec, val)
        c+=1

    plt.plot(n.voltage_track)
    plt.title("Voltage Track")
    plt.xlabel("Time (msec)")
    plt.ylabel("Membrane Potential (mvolt)")
    plt.show()

    plt.plot(n.spike_track)
    plt.title("Spike Track")
    plt.xlabel("Time (msec)")
    plt.ylabel("Spike Events (a.u.)")
    plt.show()

    plt.plot(n.wadex_track)
    plt.title("Adaptation Conductance Track")
    plt.xlabel("Time (msec)")
    plt.ylabel("Adaptation Conductance (nsiem)")
    plt.show()
    