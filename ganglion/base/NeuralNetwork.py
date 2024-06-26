from SynapticGroup import BaseSynapticGroup, PairSTDPSynapticGroup, TripletSTDPSynapticGroup, DASTDPSynapticGroup, InhibitorySynapticGroup, LatiSynapticGroup
from NeuralGroup import SensoryNeuralGroup
import numpy as np
import threading
import ntpath
import os
from units import *

import time


class NeuralNetwork:
    """
    Defines a network of neurons, more precisely, a network of neural groups, defined in a "liquid" way: i.e., the groups/mathematical order of operations are not
    set in stone and are calculated at the time of calling run_order
    """
    def __init__(self, groups, name, tki, track_activity=False, tracking_time=5*msec):
        self.tki = tki  # the timekeeper instance shared amongst the entire network
        self.name = name  # the name of this neural network
        self.neural_groups = groups  # the neural groups within this network
        self.synapses = [] # the synaptic groups within this network
        self.track_activity = track_activity  # True if we want to track the network's activity
        self.a = 0  # current network activity (Hz)
        self.tracking_time = tracking_time  # spike count tracking duration
        self.n = self.tracking_time / self.tki.dt()  # number of spike count samples to track
    
    def g(self, tag):
        """
        Return the first neuron group with the specified tag
        """
        for g in self.neural_groups:
            if g.name == tag:
                return g

        raise ValueError("There is no NeuralGroup with the name %s in this NeuralNetwork." % tag)

    def get_size_of_left_neighbor_group(self, tag):
        """
        Get the size of the neighboring group to the left of the given neural group. this is for visual layour in Viz
        """
        # get the neural group we were given the tag for
        g = self.g(tag)

        # get the column position of this group in the layer
        c = g.viz_layer_pos[1]

        # find the group in the network with c = c - 1
        for g2 in self.neural_groups:
            if g2.viz_layer_pos[1] == c - 1:
                return g2.field_shape[1]
        
        # if this is the first one, then there is zero gap
        return 0

    def fully_connect(self, g1_tag, g2_tag, trainable=True, w_i=None, stdp_params=None, syn_params=None, minw=0.01, maxw=0.9, skip_one_to_one=False, s_type='pair', loaded_weights=None, enable_wd=False):
        """
        Connect all of the neurons in g1 to every neuron in g2 with a given probability.
        if skip_one_to_one is set to True, then for each neuron, n1 in g1, connect n1 to each neuron, n2 in g2, 
        ONLY IF n1 DOES NOT have an axon projecting to n2. This is useful for lateral inhibition.
        """
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        wm = np.ones((g1.shape[0], g2.shape[0]), dtype=float)
        if skip_one_to_one:
            np.fill_diagonal(wm, 0.0)

        if s_type == 'pair':
            s = PairSTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights, enable_wd=enable_wd)
        elif s_type == 'triplet':
            s = TripletSTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights, enable_wd=enable_wd)
        elif s_type == 'base':
            s = BaseSynapticGroup(g1, g2, self.tki, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights, enable_wd=enable_wd)
        elif s_type == 'da':
            s = DASTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights)
        elif s_type == 'inhib':
            s = InhibitorySynapticGroup(g1, g2, self.tki, initial_w=w_i, w_rand_min=minw, w_rand_max=maxw, syn_params=syn_params, weight_multiplier=wm, loaded_weights=loaded_weights, enable_wd=enable_wd)
        else:
            raise RuntimeError("%s is not a valid synapse time, must be pair, triplet, or base" % (s_type))

        # store the new synaptic group into memory
        self.synapses.append(s)
    
    def one_to_one_connect(self, g1_tag, g2_tag, trainable=True, w_i=None, stdp_params=None, syn_params=None, minw=0.01, maxw=0.9, s_type='pair', loaded_weights=None, enable_wd=False):
        """
        Connect all of the neurons in g1 to the neurons in g2 at the same position as they are in g1.
        """
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        if g1.shape != g2.shape:
            raise ValueError("The shape of the first neural group must be the same shape of the second neural group but are : %s and %s" % (str(g1.shape), str(g2.shape)))

        wm = np.zeros((g1.shape[0], g2.shape[0]), dtype=float)
        
        # one-to-one mapping
        np.fill_diagonal(wm, 1.0)

        if s_type == 'pair':
            s = PairSTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights, enable_wd=enable_wd)
        elif s_type == 'triplet':
            s = TripletSTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights, enable_wd=enable_wd)
        elif s_type == 'base':
            s = BaseSynapticGroup(g1, g2, self.tki, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights, enable_wd=enable_wd)
        elif s_type == 'da':
            s = DASTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights, enable_wd=enable_wd)
        elif s_type == 'inhib':
            s = InhibitorySynapticGroup(g1, g2, self.tki, initial_w=w_i, w_rand_min=minw, w_rand_max=maxw, syn_params=syn_params, weight_multiplier=wm, loaded_weights=loaded_weights, enable_wd=enable_wd)
        elif s_type == 'lati':
            s = LatiSynapticGroup(g1, g2)
        else:
            raise RuntimeError("%s is not a valid synapse time, must be pair, triplet, or base" % (s_type))

        # store the new synaptic group into memory
        self.synapses.append(s)
    
    def local_connect(self, g1_tag, g2_tag, kernel, rstride, cstride, trainable=True, w_i=None, stdp_params=None, syn_params=None, minw=0.01, maxw=0.9, s_type='pair', loaded_weights=None, share_weights=False, enable_wd=False):
        """
        Connect the first group to the second group in a locally-connected pattern. This is equivalent to a LocallyConnected Layer in Keras
        """
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        wm = np.zeros((g1.shape[0], g2.shape[0]), dtype=float)
        # patch mapping
        a_rows = g1.field_shape[0]
        a_cols = g1.field_shape[1]
        b_rows = g2.field_shape[0]
        b_cols = g2.field_shape[1]
        kernel_rows = kernel[0]
        kernel_cols = kernel[1]

        # calculate number of strides that will be performed
        try:
            num_strides_col = (a_cols - kernel_cols) / cstride + 1
        except ZeroDivisionError:
            num_strides_col = 1.0
        try:
            num_strides_row = (a_rows - kernel_rows) / rstride + 1
        except ZeroDivisionError:
            num_strides_row = 1.0

        # check if given parameters are valid
        if a_rows < b_rows:
            raise ValueError("The number of rows in matrix B must be greater than or equal to the number of rows in matrix A")
        if a_cols < b_cols:
            raise ValueError("The number of columns in matrix B must be greater than or equal to the number of columns in matrix A")
        if not num_strides_col.is_integer():
            raise ValueError("Uneven column stride: %s" % str(num_strides_col))
        if not num_strides_row.is_integer():
            raise ValueError("Uneven row stride: %s" % str(num_strides_row))
        # convert our number of strides to integers, we couldn't do this before, since we wanted to check if the
        # strides were valid based on whether or not they evaluated to whole numbers
        num_strides_row = int(num_strides_row)
        num_strides_col = int(num_strides_col)
        # based on the number of strides, calculate the expected shape of group 2
        expected_shape = (num_strides_row, num_strides_col)
        if g2.field_shape != expected_shape:
            raise ValueError("Given shape of matrix B does not match expected shape: expected shape :: %s :: given shape :: %s" %(str(expected_shape), str(g2.field_shape)))

        # maps indices from group field shapes to indices in wm
        w_a_keys = np.reshape(np.arange(0, a_rows * a_cols, 1), (a_rows, a_cols))
        w_b_keys = np.reshape(np.arange(0, b_rows * b_cols, 1), (b_rows, b_cols))

        row_maps = []  # NOTE: Experimental for weight sharing

        # fill wm in the appropriate locations
        for r_s in range(num_strides_row):
            r_index = r_s * rstride  # row index in A
            for c_s in range(num_strides_col):
                c_index = c_s * cstride  # col index in A

                row_map = w_a_keys[r_index: r_index + kernel_rows, c_index: c_index + kernel_cols].flatten()
                row_maps.append(row_map)  # NOTE: Experimental for weight sharing
                
                wm[row_map, w_b_keys[r_s, c_s]] = 1.0
        row_maps = np.array(row_maps)  # NOTE: Experimental for weight sharing
        # define the synaptic group
        if s_type == 'pair':
            s = PairSTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights, localized_normalization=True, share_weights=share_weights, enable_wd=enable_wd)
            # NOTE: Experimental for weight sharing
            if share_weights:
                s.set_row_maps(row_maps)
        elif s_type == 'triplet':
            s = TripletSTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights, localized_normalization=True, enable_wd=enable_wd)
        elif s_type == 'base':
            s = BaseSynapticGroup(g1, g2, self.tki, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights, localized_normalization=True, enable_wd=enable_wd)
        elif s_type == 'da':
            s = DASTDPSynapticGroup(g1, g2, self.tki, trainable=trainable, stdp_params=stdp_params, syn_params=syn_params, w_rand_min=minw, w_rand_max=maxw, weight_multiplier=wm, initial_w=w_i, loaded_weights=loaded_weights, localized_normalization=True, enable_wd=enable_wd)
        elif s_type == 'inhib':
            s = InhibitorySynapticGroup(g1, g2, self.tki, initial_w=w_i, w_rand_min=minw, w_rand_max=maxw, syn_params=syn_params, weight_multiplier=wm, loaded_weights=loaded_weights, localized_normalization=True, enable_wd=enable_wd)
        else:
            raise RuntimeError("%s is not a valid synapse type, must be pair, triplet, or base" % (s_type))

        # store the new synaptic group into memory
        self.synapses.append(s)

        return s  # NOTE: Not sure if this affects anything

    def multi_fully_connect(self, pre_tags, post_tags, trainable=True, w_i=None, stdp_params=None, syn_params=None, minw=0.01, maxw=0.9, skip_one_to_one=False, s_type='pair', loaded_weights=None, enable_wd=False):
        """
        Convenience function for connecting all groups in pre_tags to all groups in post_tags in the same way, with the same parameters
        """
        for pre_tag in pre_tags:
            for post_tag in post_tags:
                self.fully_connect(pre_tag, post_tag, trainable=trainable, w_i=w_i, stdp_params=stdp_params, syn_params=syn_params, minw=minw, maxw=maxw, skip_one_to_one=skip_one_to_one, s_type=s_type, loaded_weights=loaded_weights, enable_wd=enable_wd)
    
    def multi_one_to_one_connect(self, pre_tags, post_tags, trainable=True, w_i=None, stdp_params=None, syn_params=None, minw=0.01, maxw=0.9, s_type='pair', loaded_weights=None, enable_wd=False):
        """
        Convenience function for connecting all groups in pre_tags to all groups in post_tags in the same way, with the same parameters
        """
        for pre_tag in pre_tags:
            for post_tag in post_tags:
                self.one_to_one_connect(pre_tag, post_tag, trainable=trainable, w_i=w_i, stdp_params=stdp_params, syn_params=syn_params, minw=minw, maxw=maxw, s_type=s_type, loaded_weights=loaded_weights, enable_wd=enable_wd)
    
    def multi_local_connect(self, pre_tags, post_tags, kernel, rstride, cstride, trainable=True, w_i=None, stdp_params=None, syn_params=None, minw=0.01, maxw=0.9, s_type='pair', loaded_weights=None, share_weights=False, enable_wd=False):
        """
        Convenience function for connecting all groups in pre_tags to all groups in post_tags in the same way, with the same parameters
        """
        for pre_tag in pre_tags:
            for post_tag in post_tags:
                self.local_connect(pre_tag, post_tag, kernel, rstride, cstride, trainable=trainable, w_i=w_i, stdp_params=stdp_params, syn_params=syn_params, minw=minw, maxw=maxw, s_type=s_type, loaded_weights=loaded_weights, share_weights=share_weights, enable_wd=enable_wd)
    
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
    
    def save_w(self, path_dir):
        """
        Saves the weight matrix for all synapses in the network. Returns True if successful
        """
        for s in self.synapses:
            g1_tag = s.pre_n.name
            g2_tag = s.post_n.name 

            fname = "%s_%s.npy" % (g1_tag, g2_tag)
            s.save_weights(os.path.join(path_dir, fname))
        return True

    def load_w(self, path_dir):
        """
        Loads all weights in the network. File names must be of the format "g1_tag"_"g2_tag".npy. Returns True on success and False otherwise
        """
        for fname in os.listdir(path_dir):
            if fname.endswith(".npy"):
                w = np.load(os.path.join(path_dir, fname))
                g1_tag = fname.split('_')[0]
                g2_tag = fname.split('_')[1].replace('.npy', '')

                for s in self.synapses:
                    if s.pre_n.name == g1_tag:
                        if s.post_n.name == g2_tag:
                            s.set_weights(w)

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
                        # assign the new spikes
                        s2.pre_fire_notify(g.spike_count)
            else:
                i = 0
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
        
        if self.track_activity:
            self.track_network_activity()

    def dopamine_puff(self, intensity, actions=None):
        """
        The actions flag currently does not do anything but may be used in the future for
        other purposes
        """
        for s in self.synapses:
            if type(s) == DASTDPSynapticGroup:
                s.apply_dopamine(intensity, actions=actions)
    
    def reset(self):
        for g in self.neural_groups:
            g.reset()
        for s in self.synapses:
            s.reset()

    def normalize_weights(self):
        for s in self.synapses:
            if s.trainable:
                s.normalize_weights()

    def decay_weights(self):
        for s in self.synapses:
            if s.trainable:
                s.decay_weights()

    def track_network_activity(self):
        """
        Tracks total network activity (spikes per number of tracking ste NOTE: This is not firing frequency, 
        in order to get the firing frequency, one must divide this value by the tracking time.
        """
        for g in self.neural_groups:
            if type(g) != SensoryNeuralGroup:
                self.a -= self.a / self.n
                self.a += g.spike_count.sum() / self.n
    
    def current_activity(self):
        return self.a / self.tracking_time
