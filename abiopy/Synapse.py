from Neuron import AdExNeuron
from units import *
from learning import dw
import numpy as np


class STDPParams:
    def __init__(self):
        self.window = 50 *msec
        self.lr_plus = 0.01
        self.lr_minus = 0.01
        self.tao_plus = 17 * msec
        self.tao_minus = 34 * msec
        self.a_plus = 0.3
        self.a_minus = -0.6


class Synapse:
    # NOTE: Add weight chaange tracking by turning self.w into a property call
    def __init__(self, id, pre_n: AdExNeuron, post_n: AdExNeuron, w: float, tki, trainable: bool=True, params: STDPParams=None):
        self.tki = tki
        self.pre_neuron = pre_n
        self.post_n = post_n
        self.w = w
        self.id = 1
        self.trainable = trainable

        self.lp = STDPParams() if params is None else params
        
        self.pre_spikes = []
        self.post_spikes = []

        self.weight_track = []
        self.track_weight = False
    
    def stdp(self):
        # if this synapse is trainable
        if self.trainable:
            # if the postsynaptic neuron has fired
            if len(self.post_spikes) > 0:
                post_spikes_copy = self.post_spikes.copy()
                for t in range(len(self.post_spikes)):
                    # get the time that the postsynaptic neuron fired
                    t_post = post_spikes_copy[t]
                    # if the current timestep is at least one window distance past the postsynaptic neuron's firing time
                    if self.tki.tick_time() - t_post >= self.lp.window:
                        pre = np.array(self.pre_spikes)
                        # calculate our dt between the presynaptic spikes and the postsynaptic spike
                        dt = t_post - pre
                        # only consider presynaptic spikes that are within the learning window
                        relevant_dt = dt[np.where(np.abs(dt) <= self.lp.window)]
                        bef = self.w
                        # update the weight of this synapse
                        for val in relevant_dt:
                            delta_w = dw(val, self.w, self.lp.tao_plus, self.lp.tao_minus, self.lp.lr_plus, self.lp.lr_minus, a_plus=self.lp.a_plus, a_minus=self.lp.a_minus)
                            self.w += delta_w
                        # print("My weight changed! : %g"%(self.w - bef))
                        # remove the postsynaptic spikes that were just evaluated from the post-spikes list
                        self.post_spikes = self.post_spikes[1:]
                        # remove all presynaptic spikes that occured before the currently evaluated postsynaptic spike, as these are now irrelevant to future spikes
                        self.pre_spikes = pre[np.where(pre >= t_post)].tolist()
