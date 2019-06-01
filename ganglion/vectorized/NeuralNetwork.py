from SynapticGroup import BaseSynapticGroup, PairSTDPSynapticGroup, TripletSTDPSynapticGroup, DASTDPSynapticGroup
from NeuralGroup import SensoryNeuralGroup
import numpy as np
import multiprocessing


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

        raise ValueError("There is no NeuralGroup with the name %s in this NeuralNetwork." % tag)

    def fully_connect(self, g1_tag, g2_tag, trainable=True, w_i=None, stdp_params=None, syn_params=None, minw=0.01, maxw=0.9, skip_one_to_one=False, s_type='pair', loaded_weights=None):
        """
        Connect all of the neurons in g1 to every neuron in g2 with a given probability.
        if skip_one_to_one is set to True, then for each neuron, n1 in g1, connect n1 to each neuron, n2 in g2, 
        ONLY IF n1 DOES NOT have an axon projecting to n2. This is useful for lateral inhibition.
        """
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        wm = np.ones((g1.shape[0], g2.shape[0]), dtype=np.float)
        if skip_one_to_one:
            np.fill_diagonal(wm, 0.0)

        if s_type == 'pair':
            s = PairSTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights)
        elif s_type == 'triplet':
            s = TripletSTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights)
        elif s_type == 'base':
            s = BaseSynapticGroup(g1, g2, self.tki, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights)
        elif s_type == 'da':
            s = DASTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights)
        else:
            raise RuntimeError("%s is not a valid synapse time, must be pair, triplet, or base" % (s_type))

        # store the new synaptic group into memory
        self.synapses.append(s)
    
    def one_to_one_connect(self, g1_tag, g2_tag, trainable=True, w_i=None, stdp_params=None, syn_params=None, minw=0.01, maxw=0.9, s_type='pair', loaded_weights=None):
        """
        Connect all of the neurons in g1 to the neurons in g2 at the same position as they are in g1.
        """
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        if g1.shape != g2.shape:
            raise ValueError("The shape of the first neural group must be the same shape of the second neural group but are : %s and %s" % (str(g1.shape), str(g2.shape)))
        
        wm = np.zeros((g1.shape[0], g2.shape[0]), dtype=np.float)
        # one-to-one mapping
        np.fill_diagonal(wm, 1.0)

        # s = SynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, stdp_form=stdp_form, loaded_weights=loaded_weights)

        if s_type == 'pair':
            s = PairSTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights)
        elif s_type == 'triplet':
            s = TripletSTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights)
        elif s_type == 'base':
            s = BaseSynapticGroup(g1, g2, self.tki, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights)
        elif s_type == 'da':
            s = DASTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights)
        else:
            raise RuntimeError("%s is not a valid synapse time, must be pair, triplet, or base" % (s_type))

        # store the new synaptic group into memory
        self.synapses.append(s)
    
    def convolve_connect(self, g1_tag, g2_tag, patch, rstride, cstride, trainable=True, w_i=None, stdp_params=None, syn_params=None, minw=0.01, maxw=0.9, s_type='pair', loaded_weights=None):
        """
        Connect the first group to the second group in a convolutional/patched pattern. NOTE: Find a better way to
        describe this
        """
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        wm = np.zeros((g1.shape[0], g2.shape[0]), dtype=np.float)
        # patch mapping
        a_rows = g1.field_shape[0]
        a_cols = g1.field_shape[1]
        b_rows = g2.field_shape[0]
        b_cols = g2.field_shape[1]
        patch_rows = patch.shape[0]
        patch_cols = patch.shape[1]

        # calculate number of strides that will be performed
        try:
            num_strides_col = (a_cols - patch_cols) / cstride + 1
        except ZeroDivisionError:
            num_strides_col = 1.0
        try:
            num_strides_row = (a_rows - patch_rows) / rstride + 1
        except ZeroDivisionError:
            num_strides_row = 1.0

        # check if given parameters are valid
        if a_rows < b_rows:
            raise ValueError("The number of rows in matrix B must be greater than or equal to the number of rows in matrix A")
        if a_cols < b_cols:
            raise ValueError("The number of columns in matrix B must be greater than or equal to the number of columns in matrix A")
        if not num_strides_col.is_integer():
            raise ValueError("Uneven column stride")
        if not num_strides_row.is_integer():
            raise ValueError("Uneven row stride")
        # convert our number of strides to integers, we couldn't do this before, since we wanted to check if the
        # strides were valid based on whether or not they evaluated to whole numbers
        num_strides_row = int(num_strides_row)
        num_strides_col = int(num_strides_col)
        # based on the number of strides, calculate the expected shape of group 2
        expected_shape = (num_strides_row, num_strides_col)
        if g2.field_shape != expected_shape:
            raise ValueError("Expected shape of matrix B does not match expected shape of matrix A")

        # maps indices from group field shapes to indices in wm
        w_a_keys = np.reshape(np.arange(0, a_rows * a_cols, 1), (a_rows, a_cols))
        w_b_keys = np.reshape(np.arange(0, b_rows * b_cols, 1), (b_rows, b_cols))

        # fill wm in the appropriate locations
        for r_s in range(num_strides_row):
            r_index = r_s * rstride  # row index in A
            for c_s in range(num_strides_col):
                c_index = c_s * cstride  # col index in A

                a_mod = w_a_keys[r_index: r_index + patch_rows, c_index: c_index + patch_cols].flatten()
                
                wm[a_mod, w_b_keys[r_s, c_s]] = 1.0

        # define the synaptic group
        # s = SynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, stdp_form=stdp_form, loaded_weights=loaded_weights)
        
        if s_type == 'pair':
            s = PairSTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights)
        elif s_type == 'triplet':
            s = TripletSTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights)
        elif s_type == 'base':
            s = BaseSynapticGroup(g1, g2, self.tki, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights)
        elif s_type == 'da':
            s = DASTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights)
        else:
            raise RuntimeError("%s is not a valid synapse time, must be pair, triplet, or base" % (s_type))

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
    
    def save_w(self, path, g1_tag, g2_tag):
        """
        Saves the weight matrix between two neural groups to the specified path. Returns True if successful
        """
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        for s in self.synapses:
            if s.pre_n == g1 and s.post_n == g2:
                s.save_weights(path)
                return True
        
        return False

    def set_trainability(self, val: bool):

        """Set the trainability of the synaptic groups in this network"""

        for s in self.synapses:
            s.trainable = val
    
    def run_order(self, group_order):
        """Runs through each neuron group in the specified order, evaluating inputs received in the last dt time step
        and generating/propagating resultant signals that would have occured in said time step"""

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
                i = 0#np.zeros(g.shape, dtype=np.float)
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

    def dopamine_puff(self, intensity, actions=None):
        for s in self.synapses:
            if type(s) == DASTDPSynapticGroup:
                s.apply_dopamine(intensity, actions = actions)
    
    def reset(self):
        for g in self.neural_groups:
            g.reset()
        for s in self.synapses:
            s.reset()

    def normalize_weights(self):
        for s in self.synapses:
            if s.trainable:
                s.normalize_weights()
