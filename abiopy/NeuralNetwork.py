from typing import List
import random
from Synapse import Synapse

class NeuralNetwork:
    def __init__(self, groups, name):
        self.name = name
        self.neural_groups = groups
        self.s: List[Synapse] = []
    
    def g(self, tag):
        """
        Return the first neuron with the specified tag
        """
        for g in self.neural_groups:
            if g.name == tag:
                return g
        
        return None

    def fully_connect(self, g1, g2, connection_probability=1.0, trainable=True, w_i=None):
        """
        Connect all of the neurons in g1 to every neuron in g2 with a given probability
        """
        for n1 in self.neural_groups[g1].n:
            for n2 in self.neural_groups[g2].n:
                if random.random() <= connection_probability:
                    s = Synapse(0, n1, n2, random.uniform(0.01, 0.1) if w_i is None else w_i, trainable=trainable)
                    n1.axonal_synapses.append(s)
                    n2.dendritic_synapses.append(s)
                    self.s.append(s)
    
    def one_to_all(self, n1, g2_tag, connection_probability=1.0, trainable=True, w_i=None):
        """
        Collect a single neuron to every neuron in g2 with a given probability
        """
        g2 = self.g(g2_tag)
        for n2 in g2:
            if random.random() <= connection_probability:
                s = Synapse(0, n1, n2, random.uniform(0.01, 0.1) if w_i is None else w_i, trainable=trainable)
                n1.axonal_synapses.append(s)
                n2.dendritic_synapses.append(s)
                self.s.append(s)

    def all_to_one(self, g1_tag, n2, connection_probability=1.0, trainable=True, w_i=None):
        """
        Connect every neuron in g1 to a single neuron with a given probability
        """
        g1 = self.g(g1_tag)
        for n1 in g1:
            if random.random() <= connection_probability:
                s = Synapse(0, n1, n2, random.uniform(0.01, 0.1) if w_i is None else w_i, trainable=trainable)
                n1.axonal_synapses.append(s)
                n2.dendritic_synapses.append(s)
                self.s.append(s)
        
    def run_order(self, group_order, timekeeper, train=True):
        # Evaluating the inputs into each neuron and generating outputs
        # loop over each NeuronGroup
        for o in group_order:
            g = self.neural_groups[o]
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
