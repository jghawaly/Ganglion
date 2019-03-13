from Neuron import SpikingNeuron
from units import *
from learning import dw


class STDPParams:
    def __init__(self):
        self.window = 5 *msec
        self.lr_plus = 0.01
        self.lr_minus = 0.01
        self.tao_plus = 10 * msec
        self.tao_minus = 10 * msec
        self.a_plus = 0.3
        self.a_miunus = -0.6

class Synapse:
    # NOTE: Add weight chaange tracking by turning self.w into a property call
    def __init__(self, id, pre_n: SpikingNeuron, post_n: SpikingNeuron, w: float=1.0, trainable: bool=True, params: STDPParams=None):
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
        if self.trainable:
            # NOTE: Need to watch this carefully, could become a potential memory leak if the pre and postsynaptic spike history is not dumped
            if len(self.post_spikes) >= 2:
                for i in range(len(self.post_spikes)-1):
                    # first postsynaptic spike time
                    t1 = self.post_spikes[i]
                    # second postsynaptic spike time
                    t2 = self.post_spikes[i+1]

                    # parse out presynaptic spikes that have an effect on the spike being evaluated
                    relevant_pre = [t for t in self.pre_spikes if t < t2]

                    # delta time between the postsynaptic spike being evaluated and all relevant presynaptic spikes
                    dt = [t1 - t for t in relevant_pre]

                    # calculate total synaptic weight change (should this be recursive?)
                    # dw_total = np.array([dw(val, self.w, 10 * msec, 10 * msec, 1.0, 1.0) for val in dt]).sum()
                    # self.w += dw_total
                    bef = self.w
                    for val in dt:
                        self.w += dw(val, self.w, self.lp.tao_plus, self.lp.tao_minus, self.lp.lr_ex, self.lp.lr_inh, a_plus=self.lp.a_plus, a_minus=self.lp.a_minus)
                    
                    # if self.post_n.group_scope == "hidden" and len(relevant_pre) > 0:
                    #     print(dt)
                    #     print("weight_changed: %g" % (self.w - bef))
                    #     print(self.w)
                    #     print()

                    # remove presynaptic spikes that occured before the spike that STDP was just evaluated on
                    self.pre_spikes = [t for t in self.pre_spikes if t > t1]

                    # remove the post synaptic spike that was just evaluated
                self.post_spikes = [self.post_spikes[-1]]
            elif len(self.post_spikes) == 1:
                # cull presynaptic spikes that are outside of our window. We need to keep spikes that come 2*window after the postsynaptic
                # spike, otherwise we may cull presynaptic spikes to the NEXT postsynaptic spike, which has not yet arrived
                self.pre_spikes = [t for t in self.pre_spikes if -2*self.lp.window <= self.post_spikes[0] - t <= self.lp.window]
            elif len(self.post_spikes) == 0:
                pass
            else:
                print("What even just happened?")
        else:
            self.pre_spikes = []
            self.post_spikes = []


        