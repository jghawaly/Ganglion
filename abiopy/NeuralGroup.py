from Neuron import SpikingNeuron, Synapse, NeuronParams
from timekeeper import TimeKeeperIterator
from units import *
import numpy as np
import random
from typing import List


class NeuralGroup:
    def __init__(self, inhib_neurons: int, excit_neurons: int, name, neuron_params=None):
        self.name = name
        self.i_num = inhib_neurons
        self.e_num = excit_neurons
        self.n_num = inhib_neurons + excit_neurons
        self.neuron_params = neuron_params

        self.n: List[SpikingNeuron] = []
        self.s: List[Synapse] = []
        self._construct()
    
    def _construct(self):
        # either load default or user-specified neuron parameters
        neuron_params = NeuronParams() if self.neuron_params is None else self.neuron_params

        for i in range(self.i_num):
            self.n.append(SpikingNeuron(SpikingNeuron.inhibitory, neuron_params, group_scope=self.name))
        for i in range(self.e_num):
            self.n.append(SpikingNeuron(SpikingNeuron.excitatory, neuron_params, group_scope=self.name))
    
    def evaluate(self, dt, current_timestep):
        for n in self.n:
            n.evaluate(dt, current_timestep)
    
    def track_vars(self, v):
        for n in self.n:
            n.tracked_vars = v
    
    def dci(self, c):
        if c.shape[0] != len(self.n):
            raise ValueError("Input array must be same length as the number of neurons in this NeuronGroup, but are %g and %g." % (c.shape[0], len(self.n)))
        else:
            for i in range(len(c)):
                self.n[i].dendritic_spikes.append({'neuron_type': SpikingNeuron.dci, 'weight': c[i]})
    
    @property
    def voltage_track(self):
        return [n.voltage_track[-1] for n in self.n]
    
    @property
    def current_track(self):
        return [n.current_track[-1] for n in self.n]
    
    @property
    def spike_track(self):
        return [n.spike_track[-1] for n in self.n]


class SquareNeuralGroup(NeuralGroup):
    def __init__(self, kernel: np.ndarray, name, neuron_params=None):
        self.kernel = kernel
        self.n_structure = []
        super().__init__(kernel.shape[0] * kernel.shape[1] - np.count_nonzero(kernel), np.count_nonzero(kernel), name, neuron_params=None)
    
    def _construct(self):
        # either load default or user-specified neuron parameters
        neuron_params = NeuronParams() if self.neuron_params is None else self.neuron_params

        for idx, i in np.ndenumerate(self.kernel):
            if i == 0:
                neuron = SpikingNeuron(SpikingNeuron.inhibitory, neuron_params, group_scope=self.name)
            else:
                neuron = SpikingNeuron(SpikingNeuron.excitatory, neuron_params, group_scope=self.name)
            self.n.append(neuron)
            self.n_structure.append({'kernel_loc': idx, "neuron": neuron})
    
    @property
    def weight_map(self):
        w_map = np.zeros(self.kernel.shape, dtype=np.float)
        for item in self.n_structure:
            w_map[item['kernel_loc']] = item['neuron'].axonal_synapses[0].w
        
        return w_map
    
    def dci(self, c):
        if c.shape != self.kernel.shape:
            raise ValueError("Input array must be same length as the number of neurons in this NeuronGroup, but are %g and %g." % (c.shape[0], len(self.n)))
        else:
            for item in self.n_structure:
                item['neuron'].dendritic_spikes.append({'neuron_type': SpikingNeuron.dci, 'weight': c[item['kernel_loc']]})
