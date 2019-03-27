from units import *
import random
import numpy as np
from typing import List, Dict
from learning import dw
from numba import jit


class AdExParams:
    def __init__(self):
        # self-defined parameters
        self.refractory_period = 2.0 * msec

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


@jit(nopython=True)
def calc_dvm(dt, sf, vm, vthr, vr, tao_m, i_syn, w, c_m):
    return dt * ((sf * np.exp((vm - vthr) / sf) - (vm - vr)) / tao_m - (i_syn - w) / c_m)


@jit(nopython=True)
def calc_dw(dt, a, v_m, v_r, w, tao_w):
    return dt * ((a * (v_m - v_r) - w) / tao_w)


@jit(nopython=True)
def calc_isyn(spike_weight, v_m, vrev, gbar, delta_t, tao_syn):
    if delta_t <= 0.0:
        return 0.0
    else:
        return spike_weight * (v_m - vrev) * gbar * np.exp(-delta_t / tao_syn)


class AdExNeuron:
    inhibitory=0  # inhibitory neuron
    excitatory=1  # excitatory neuron
    # spike_forcer=2  # forces a spike in the neurons that it projects too
    desi=3  # direct excitatory spike injection
    disi=4  # direct inhibitory spike injection
    dci=5  # direct current injection
    def __init__(self, neuron_type, params: AdExParams, tki, group_scope="single"):
        self.n_type = neuron_type

        # global timekeeper reference
        self.tki = tki

        # refractory period
        self.refractory_period = params.refractory_period

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
    
    def filter_spikes(self):
        """
        Only keep dendritic spikes that are withing the spike evaluation window
        """
        spike_times= np.array(self.dendritic_spike_times)
        spike_weights = np.array(self.dendritic_spike_weights)
        spike_types = np.array(self.dendritic_spike_types)

        # calculate time-since-spike for all dendritic spikes
        tss = self.tki.tick_time() - spike_times

        windowed_indices = np.where(tss <= self.spike_window)

        self.dendritic_spike_times = spike_times[windowed_indices].tolist()
        self.dendritic_spike_weights = spike_weights[windowed_indices].tolist()
        self.dendritic_spike_types = spike_types[windowed_indices].tolist()
    
    def psp(self):
        """
        Calculate the total input current coming in on the dendritic synapses
        """
        # we only want to keep and evaluate dendritic spikes within the window of interest
        self.filter_spikes()

        # track the total synaptic input current
        i_total = 0.0
        for i in range(len(self.dendritic_spike_times)):
            spike_time = self.dendritic_spike_times[i]
            spike_weight = self.dendritic_spike_weights[i]
            spike_type = self.dendritic_spike_types[i]

            # calculate time since this spike occured
            delta_t = self.tki.tick_time() - spike_time

            if spike_type == self.__class__.excitatory:
                vrev = self.vrev_e 
                gbar = self.gbar_e
                # isyn = spike_weight * (self.v_m - vrev) * gbar * np.exp(-delta_t / self.tao_syn) * np.heaviside(delta_t, 0.0)
                isyn = calc_isyn(spike_weight, self.v_m, vrev, gbar, delta_t, self.tao_syn)
            elif spike_type == self.__class__.inhibitory:
                vrev = self.vrev_i
                gbar = self.gbar_i
                # isyn = spike_weight * (self.v_m - vrev) * gbar * np.exp(-delta_t / self.tao_syn) * np.heaviside(delta_t, 0.0)
                isyn = calc_isyn(spike_weight, self.v_m, vrev, gbar, delta_t, self.tao_syn)
            elif spike_type == self.__class__.desi:
                vrev = self.vrev_e
                gbar = self.gbar_e
                # isyn = spike_weight * (self.v_m - vrev) * gbar * np.exp(-delta_t / self.tao_syn) * np.heaviside(delta_t, 0.0)
                isyn = calc_isyn(spike_weight, self.v_m, vrev, gbar, delta_t, self.tao_syn)
            elif spike_type == self.__class__.disi:
                vrev = self.vrev_i
                gbar = self.gbar_i
                # isyn = spike_weight * (self.v_m - vrev) * gbar * np.exp(-delta_t / self.tao_syn) * np.heaviside(delta_t, 0.0)
                isyn = calc_isyn(spike_weight, self.v_m, vrev, gbar, delta_t, self.tao_syn)
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

    def in_refractory(self):
        """
        Check if this neuron is in its refractory period
        """
        if self.last_spike_time == 0.0:
            return False
        else:
            if (self.tki.tick_time() - self.last_spike_time) > self.refractory_period:
                return False
            else:
                return True

    def evaluate(self):
        """
        Update the state of the neuron.
        """
        output = self.v_m
        self.spiked = 0

        if not self.in_refractory():
            # get input current
            i_syn = self.psp()

            # update membrane potential
            # dv_m = self.tki.dt() * ((self.sf * np.exp((self.v_m - self.v_thr) / self.sf) - (self.v_m - self.v_r)) / self.tao_m - (self.i_syn - self.w) / self.c_m)
            dvm = calc_dvm(self.tki.dt(), self.sf, self.v_m, self.v_thr, self.v_r, self.tao_m, i_syn, self.w, self.c_m)
            self.v_m += dvm

            # update adaptation parameter
            # dw = self.tki.dt() * ((self.a * (self.v_m - self.v_r) - self.w) / self.tao_w)
            dw = calc_dw(self.tki.dt(), self.a, self.v_m, self.v_r, self.w, self.tao_w)
            self.w += dw
            
            # used for tracking the input charge
            q_total = 0.0

            output = self.v_m
            
            # check if ready to fire
            if self.v_m >= self.v_thr:
                self.fire()
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
        if not self.in_refractory():
            self.dendritic_spike_times.append(spike_descriptor['timestep']) 
            self.dendritic_spike_weights.append(spike_descriptor['weight']) 
            self.dendritic_spike_types.append(spike_descriptor['neuron_type']) 

    def fire(self):
        """
        Propagate spike to axonal synapses and reset the current neuron to its post-spike state
        """
        # update some tracked parameters
        self.last_spike_time = self.tki.tick_time()
        self.spiked = 1
        
        for synapse in self.axonal_synapses:
            # propagate the spike across the outgoing axonal synapse
            synapse.post_n.add_spike({'neuron_type': self.n_type, 'weight': synapse.w, 'timestep': self.last_spike_time}) 
            # notify the axonal synapse that its presynaptic neuron fired at this time step (for Hebbian learning)
            synapse.pre_spikes.append(self.last_spike_time)
        
        for synapse in self.dendritic_synapses:
            # notify all dendritic synapses that the postsynaptic neuron fired at this time step (for Hebbian learning)
            synapse.post_spikes.append(self.last_spike_time)

        # reset the neuron's membrane potential to its hyperpolarized value
        self.reset()
