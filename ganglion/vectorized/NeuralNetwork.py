from SynapticGroup import SynapticGroup
from NeuralGroup import SensoryNeuralGroup
from parameters import STDPParams
import numpy as np


class NeuralNetwork:
    """
    Defines a network of neurons, more precisely, a network of neural groups, defined in a "liquid" way: i.e., the groups/mathematical order of operations are not
    set in stone and are calculated at the time of calling run_order
    """
    def __init__(self, groups, name, tki):
        self.tki = tki  # the timekeeper instance shared amongst the entire network
        self.name = name  # the name of this neural network
        self.neural_groups = groups  # the neural groups within this network
        self.synapses = [] # the synaptic groups within this network
    
    def g(self, tag):
        """
        Return the first neuron group with the specified tag
        """
        for g in self.neural_groups:
            if g.name == tag:
                return g

        return None

    def fully_connect(self, g1_tag, g2_tag, trainable=True, w_i=None, stdp_params=None, minw=0.01, maxw=0.9, skip_one_to_one=False):
        """
        Connect all of the neurons in g1 to every neuron in g2 with a given probability.
        if skip_one_to_one is set to True, then for each neuron, n1 in g1, connect n1 to each neuron, n2 in g2, 
        ONLY IF n1 DOES NOT have an axon projecting to n2. This is useful for lateral inhibition.
        """

        if stdp_params is None:
            stdp_params = STDPParams()
        
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        wm = np.ones((g1.shape[0], g2.shape[0]), dtype=np.float)
        if skip_one_to_one:
            np.fill_diagonal(wm, 0.0)

        s = SynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i)

        # store the new synaptic group into memory
        self.synapses.append(s)
    
    def one_to_one_connect(self, g1_tag, g2_tag, trainable=True, w_i=None, stdp_params=None, minw=0.01, maxw=0.9):
        """
        Connect all of the neurons in g1 to the neurons in g2 at the same position as they are in g1.
        """
        if stdp_params is None:
            stdp_params = STDPParams()
        
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        if g1.shape != g2.shape:
            raise ValueError("The shape of the first neural group must be the same shape of the second neural group but are : %s and %s" % (str(g1.shape), str(g2.shape)))

        wm = np.zeros((g1.shape[0], g2.shape[0]), dtype=np.float)
        np.fill_diagonal(wm, 1.0)

        s = SynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i)

        # store the new synaptic group into memory
        self.synapses.append(s)

    def get_w_between_g_and_g(self, g1_tag, g2_tag):
        """
        Returns the weight matrix between two neural groups
        """
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        for s in self.synapses:
            if s.pre_n == g1 and s.post_n == g2:
                return s.w 
        
        return None
    
    def get_w_between_g_and_n(self, g1_tag, g2_tag, n2_index):
        """
        Returns the weight matrix between two neural groups
        """
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        for s in self.synapses:
            if s.pre_n == g1 and s.post_n == g2:
                w = s.w 
                return np.reshape(w[:, n2_index].copy(), g1.field_shape)
        
        return None
    
    def set_trainability(self, val: bool):
        """
        Set the trainability of the synaptic groups in this network
        """
        for s in self.synapses:
            s.trainable = val
    
    def run_order(self, group_order):
        """
        Runs through each neuron group in the specified order, evaluating inputs received in the last dt time step
        and generating/propagating resultant signals that would have occured in said time step
        """
        # loop over each NeuronGroup
        for o in group_order:
            g = self.g(o)

            # sensory neural groups are treated differently since they do not have any input synapses
            if type(g) == SensoryNeuralGroup:
                # send out spikes to outgoing synapses
                for s2 in self.synapses:
                    # if this neuron group is the presynaptic group to this synapse group
                    if s2.pre_n == g:
                        # roll the synaptic spike history back a step and assign the new spikes
                        # s2.roll_history_and_assign(g.spike_count) 
                        s2.pre_fire_notify(g.spike_count)
            else:
                i = np.zeros(g.shape, dtype=np.float)
                # calculate total input current
                for s in self.synapses:
                    # if this synapse group is presynaptic, then calculate the current coming across it and run that current through the current neural group
                    if s.post_n == g:
                        i += s.calc_isyn()
                g.run(i)
                # send out spikes to outgoing synapses
                for s in self.synapses:
                    if s.pre_n == g:        
                        s.pre_fire_notify(g.spike_count)
                    if s.post_n == g:
                        s.post_fire_notify(g.spike_count)
            # else:
            #     for s in self.synapses:
            #         # if this synapse group is presynaptic, then calculate the current coming across it and run that current through the current neural group
            #         if s.post_n == g:
            #             i = s.calc_isyn()
            #             g.run(i)
                        
            #             # send out spikes to outgoing synapses
            #             for s2 in self.synapses:
            #                 # if this neuron group is the presynaptic group to this synapse group
            #                 if s2.pre_n == g:
            #                     # roll the synaptic spike history back a step and assign the new spikes
            #                     # s2.roll_history_and_assign(g.spike_count)
            #                     s2.pre_fire_notify(g.spike_count)
            #                 else:
            #                     s2.post_fire_notify(g.spike_count)
