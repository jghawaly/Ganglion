from typing import List
import random
from Synapse import Synapse
from NeuralGroup import NeuralGroup, StructuredNeuralGroup
import numpy as np

class NeuralNetwork:
    def __init__(self, groups, name):
        self.name = name
        self.neural_groups = groups
        self.s: List[Synapse] = []
    
    def g(self, tag):
        """
        Return the first neuron group with the specified tag
        """
        for g in self.neural_groups:
            if g.name == tag:
                return g

        return None

    def fully_connect(self, g1_tag, g2_tag, connection_probability=1.0, trainable=True, w_i=None, skip_self=False):
        """
        Connect all of the neurons in g1 to every neuron in g2 with a given probability
        if skip_self is set to True, then for each neuron, n1 in g1, connect n1 to each neuron, n2 in g2, 
        ONLY IF n2 DOES NOT have an axon projecting to n1. This is useful for lateral inhibition.
        """
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        for n1 in g1.n:
            for n2 in g2.n:
                can_connect = True

                if skip_self:
                    # set can_connect to False if any of the axonal synapses on n2 project to n1
                    for syn in n2.axonal_synapses:
                        if syn.post_n == n1:
                            can_connect = False
                        if not can_connect:
                            break
                
                # if we are allowed, connect n1 to n2
                if can_connect:
                    if random.random() <= connection_probability:
                        s = Synapse(0, n1, n2, random.uniform(0.1, 0.5) if w_i is None else w_i, trainable=trainable)
                        n1.axonal_synapses.append(s)
                        n2.dendritic_synapses.append(s)
                        self.s.append(s)
    
    def one_to_all(self, n1, g2_tag, connection_probability=1.0, trainable=True, w_i=None):
        """
        Connect a single neuron to every neuron in g2 with a given probability
        """
        g2 = self.g(g2_tag)
        for n2 in g2.n:
            if random.random() <= connection_probability:
                s = Synapse(0, n1, n2, random.uniform(0.1, 0.5) if w_i is None else w_i, trainable=trainable)
                n1.axonal_synapses.append(s)
                n2.dendritic_synapses.append(s)
                self.s.append(s)

    def all_to_one(self, g1_tag, n2, connection_probability=1.0, trainable=True, w_i=None):
        """
        Connect every neuron in g1 to a single neuron with a given probability
        """
        g1 = self.g(g1_tag)
        for n1 in g1.n:
            if random.random() <= connection_probability:
                s = Synapse(0, n1, n2, random.uniform(0.1, 0.5) if w_i is None else w_i, trainable=trainable)
                n1.axonal_synapses.append(s)
                n2.dendritic_synapses.append(s)
                self.s.append(s)
    
    def one_to_one(self, g1_tag, g2_tag, connection_probability=1.0, trainable=True, w_i=None):
        """
        Connect each neuron in g1 to a single neuron in g2 corresponding to the same location in each neuron group
        """
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        if g1.shape == g2.shape:
            for idx, _ in np.ndenumerate(np.zeros(g1.shape)):
                # this is kind of hacky
                if g1.__class__ == StructuredNeuralGroup:
                    n1 = g1.neuron(idx)
                elif g1.__class__ == NeuralGroup:
                    n1 = g1.n[idx[0]]
                if g2.__class__ == StructuredNeuralGroup:
                    n2 = g2.neuron(idx)
                elif g2.__class__ == NeuralGroup:
                    n2 = g2.n[idx[0]]

                if random.random() <= connection_probability:
                    s = Synapse(0, n1, n2, random.uniform(0.1, 0.5) if w_i is None else w_i, trainable=trainable)
                    n1.axonal_synapses.append(s)
                    n2.dendritic_synapses.append(s)
                    self.s.append(s)
        else:
            raise ValueError("NeuronGroups g1 and g2 must have the same shape")
        
    def run_order(self, group_order, timekeeper, train=True):
        # Evaluating the inputs into each neuron and generating outputs
        # loop over each NeuronGroup
        for o in group_order:
            g = self.g(o)
            # loop over every Neuron in this NeuronGroup
            g.evaluate(timekeeper.dt(), timekeeper.tick_time())
        
        # update the dendritic spike chains
        # loop over each NeuronGroup
        # for o in group_order:
        #     g = self.neural_groups[o]
        #     # loop over every Neuron in this NeuronGroup
        #     for n in g.n:
        #         n.update()
        
        if train:
            # update synaptic weights
            # loop over every synapse in the network
            for s in self.s:
                s.stdp()
