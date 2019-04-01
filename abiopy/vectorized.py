import numpy as np
from timekeeper import TimeKeeperIterator
from units import *
from utils import poisson_train
from numba import roc, njit


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
        self.vrev_e = 0.0 * mvolt
        self.vrev_i = -75.0 * mvolt
        # self.spike_window = 20.0 * msec


class SynapseParams:
    def __init__(self):
        self.window = 50 *msec
        self.lr_plus = 0.01
        self.lr_minus = 0.01
        self.tao_plus = 17 * msec
        self.tao_minus = 34 * msec
        self.a_plus = 0.3
        self.a_minus = -0.6
        self.tao_syn = 5.0 * msec  # this probably needs to be 10 msec
        self.spike_window = 20.0 * msec


class STDPParams:
    def __init__(self):
        self.a2_plus = 5.0e-10
        self.a3_plus = 6.2e-3
        self.a2_minus = 7.0e-3
        self.a3_minus = 2.3e-4
        self.tao_x = 101.0 * msec
        self.tao_y = 125.0 * msec
        self.tao_plus = 16.8 * msec
        self.tao_minus = 33.7 * msec
        self.eps = 0.33
        self.stdp_window = 20.0 * msec


class NeuralGroup:
    """
    This class is a base template containing only items that are common among most neuron models. It 
    should be overriden, and does not run on its own.
    """
    def __init__(self, n_type: np.ndarray, name: str, tki: TimeKeeperIterator):
        self.name = name  # the name of this neuron
        self.tki = tki  # the timekeeper isntance that is shared amongst the entire network
        
        # calculate the area of the neural group's geometry
        try:
            area = n_type.shape[0] * n_type.shape[1]
        except IndexError:
            area = n_type.shape[0]
        
        # this holds the given neuron identity array, and also defines the geometry of the group.
        # 0's represent the location of inhibitory neurons and 1's represent the location of excitatory neurons
        self.n_type = n_type  
        self.shape = n_type.shape  # shape/geometry of this neural group
        self.num_excitatory = np.count_nonzero(n_type)  # number of excitatory neurons in this group
        self.num_inhibitory = area - np.count_nonzero(n_type)  # number of inhibitory neurons in this group
        self.tracked_vars = []  # variables to be tracked throughout the course of evaluation

    def run(self):
        """
        Update the state of the neurons.

        This method should be overriden for different neuron models
        """
        return None


class SensoryNeuralGroup(NeuralGroup):
    """
    This class defines a group of "neurons" that are not really neurons, they simple send "spikes"
    when the user tells them too
    """
    def __init__(self, n_type: np.ndarray, name: str, tki: TimeKeeperIterator, params: AdExParams):
        super().__init__(n_type, name, tki)

        
        self.spike_count = np.zeros(self.shape, dtype=np.int)  # holds the NUMBER OF spikes that occured in the last evaluated time window
        self.v_spike = np.full(self.shape, params.v_spike)  # spike potential

        # construct reversal potential matrix
        self.v_rev = np.zeros(n_type.shape, dtype=np.float)
        self.v_rev[np.where(self.n_type==0)] = params.vrev_i
        self.v_rev[np.where(self.n_type==1)] = params.vrev_e

        # construct gbar matrix
        self.gbar = np.zeros(n_type.shape, dtype=np.float)
        self.gbar[np.where(self.n_type==0)] = params.gbar_i
        self.gbar[np.where(self.n_type==1)] = params.gbar_e

        # lists that will contain the membrane voltage track
        self.v_m_track = []
        self.spike_track = []
    
    def run(self, spike_count):
        """
        Manually "fire" the neurons at the given input
        """
        # check to make sure the input has the same dimensions as the group's shape
        if spike_count.shape != self.shape:
            raise ValueError("Input spike matrix should be the same shape as the neuron matrix but are : %s and %s" % (str(spike_count.shape), str(self.shape)))
        
        self.spike_count = spike_count

        output = np.zeros(self.shape, dtype=np.float)

        output[np.where(self.spike_count > 0)] = self.v_spike[np.where(self.spike_count > 0)]  # generate spikes where they are requested

        if "v_m" in self.tracked_vars:
            self.v_m_track.append(output.copy())
        if "spike" in self.tracked_vars:
            self.spike_track.append(spike_count.copy())

        return output


class AdExNeuralGroup(NeuralGroup):
    """
    This class defines a group of Adaptive Exponential Integrate and Fire Neurons, as described by Brette and Gerstner (2005)
    """
    def __init__(self, n_type: np.ndarray, name: str, tki: TimeKeeperIterator, params: AdExParams):
        super().__init__(n_type, name, tki)

        # custom parameters
        self.refractory_period = np.full(self.shape, params.refractory_period)  # refractory period for these neurons
        self.spiked = np.zeros(self.shape, dtype=np.int)  # holds boolean array of WHETHER OR NOT a spike occured in the last call to run() (Note: NOT THE LAST TIME STEP)
        self.spike_count = np.zeros(self.shape, dtype=np.int) # holds the NUMBER OF spikes that occured in the last evaluated time window
        self.last_spike_time = np.zeros(self.shape, dtype=np.float)  # holds the TIMES OF last spike for each neuron
        self.last_spike_count_update = 0.0  # holds the time at which the spike count array was last updated

        # construct reversal potential matrix
        self.v_rev = np.zeros(n_type.shape, dtype=np.float)  # reversal potential for any outgoing synapses from these neurons
        self.v_rev[np.where(self.n_type==0)] = params.vrev_i
        self.v_rev[np.where(self.n_type==1)] = params.vrev_e

        # construct gbar matrix
        self.gbar = np.zeros(n_type.shape, dtype=np.float)  # conductance of any outgoing synapses from these neurons
        self.gbar[np.where(self.n_type==0)] = params.gbar_i 
        self.gbar[np.where(self.n_type==1)] = params.gbar_e

        # Parameters from Brette and Gerstner (2005).
        self.v_r = np.full(self.shape, params.v_r)  # rest potential
        self.v_m = np.full(self.shape, params.v_m)  # membrane potential
        self.v_spike = np.full(self.shape, params.v_spike)  # spike potential
        self.w = np.full(self.shape, params.w)  # conductance adaptation parameter for AdEx model
        self.v_thr = np.full(self.shape, params.v_thr)  # spike threshold potential
        self.sf = np.full(self.shape, params.sf)  # slope factor for AdEx model
        self.tao_m = np.full(self.shape, params.tao_m)  # membrane time constant
        self.c_m = np.full(self.shape, params.c_m)  # membrane capacitance
        self.a = np.full(self.shape, params.a)  # a parameter for AdEx model
        self.b = np.full(self.shape, params.b)  # b parameter for AdEx model
        self.tao_w = np.full(self.shape, params.tao_w)  # decay time constant for conductance adaptation parameter for AdEx model

        # parameter tracks
        self.v_m_track = []
        self.isyn_track = []
        self.spike_track = []

    def not_in_refractory(self):
        """
        return mask of neurons that ARE NOT in refractory period
        """
        ir = np.ones(self.shape, dtype=np.int)
        # set zeros for neurons in the refractory period
        ir[np.where((self.tki.tick_time() - self.last_spike_time) < self.refractory_period)] = 0
        # if last_spike_time is 0.0, then the neuron has never fired yet, so we want to exclude this
        ir[np.where(self.last_spike_time == 0.0)] = 1
        
        return ir

    def run(self, i_syn):
        """
        Update the state of the neurons via the Adaptive Exponential Integrate and Fire neuron model
        """
        # if we are at a new time step since evaluating the neurons, then clear the spike count matrices
        if self.last_spike_count_update != self.tki.tick_time():
            self.spike_count = np.zeros(self.shape, dtype=np.int)
        
        self.spiked = np.zeros(self.shape, dtype=np.int)

        # mask of neurons not in refractory period
        refrac = self.not_in_refractory()

        # calculate change in membrane potential for neurons not in refractory period
        dvm = self.tki.dt() * ((self.sf * np.exp((self.v_m - self.v_thr) / self.sf) - (self.v_m - self.v_r)) / self.tao_m - (i_syn - self.w) / self.c_m) * refrac

        # update membrane potential
        self.v_m += dvm

        # update adaptation parameter
        dw = self.tki.dt() * ((self.a * (self.v_m - self.v_r) - self.w) / self.tao_w) * refrac
        self.w += dw

        # find indices of neurons that have fired
        self.spiked = np.where(self.v_m >= self.v_thr)
        # add a new spike to the spike count for each neuron that fired
        self.spike_count[self.spiked] += 1
        # update the time at which the spike count array was modified
        self.last_spike_count_update = self.tki.tick_time()
        # modify the last-spike-time for each neuron that fired
        self.last_spike_time[self.spiked] = self.tki.tick_time()

        # this is a copy of the membrane potential matrix
        output = self.v_m.copy()

        # change the output to the spike voltage for each neuron that fired. Note: this does not affect the actual v_m array, just a copy of it, because
        # in this neuron model, the voltage of a spike does not really have any specific meaning, rather, it is the time of the spikes that matter
        output[self.spiked] = self.v_spike[self.spiked]

        # change the actual membrane voltage to the resting potential for each neuron that fired
        self.v_m[self.spiked] = self.v_r[self.spiked]

        # if we are tracking any variables, then append them to their respective lists, Note: This can use a lot of memory and cause slowdowns, so only do this when absolutely necessary
        if "v_m" in self.tracked_vars:
            self.v_m_track.append(output.copy())
        if "i_syn" in self.tracked_vars:
            self.isyn_track.append(i_syn.copy())
        if "spike" in self.tracked_vars:
            self.spike_track.append(self.spike_count.copy())

        return output

@njit
def fast_row_roll(val, assignment):
    """
    A JIT compiled method for roll the values of all rows in a givenm matrix down by one, and
    assigning the first row to the given assignment
    """
    val[1:,:] = val[0:-1,:]
    val[0] = assignment
    return val

class SynapticGroup:
    """
    Defines a groups of synapses connecting two groups of neurons
    """
    def __init__(self, pre_n: AdExNeuralGroup, post_n: AdExNeuralGroup, tki, initial_w: float=None, w_rand_min: float=0.0, w_rand_max: float=1.0, trainable: bool=True, syn_params: SynapseParams=None, stdp_params: STDPParams=None):
        self.synp = SynapseParams() if syn_params is None else syn_params  # synapse parameters are default if None is given
        self.stdpp = STDPParams() if stdp_params is None else stdp_params  # STDP parameters are default if None is given
        self.tki = tki  # reference to timekeeper object that is shared amongst the entire network
        self.pre_n = pre_n  # presynaptic neural group
        self.post_n = post_n  # postsynaptic neural group
        self.trainable = trainable  # True if this synaptic group is trainable, False otherwise

        self.m = pre_n.shape[0]  # number of neurons in presynaptic group
        self.n = post_n.shape[0]  # number of neuronsin postsynaptic group
        self.num_synapses = self.m * self.n  # nunmber of synapses to be generated
        self.w_rand_min = w_rand_min  # minimum weight for weight-initialization via uniform distribution
        self.w_rand_max = w_rand_max  # maximum weight for weight-initialization via uniform distribution
        self.initial_w = initial_w  # initial weight value for static weight initialization
        self.w = self.construct_weights()  # construct weight matrix
        
        self.pre_spikes = []
        self.post_spikes = []

        self.num_histories = int(self.synp.spike_window / self.tki.dt())  # number of discretized time bins that we will keep track of presynaptic spikes in
        self.history = np.zeros((self.num_histories, self.m, self.n), dtype=np.float)  # A num_histories * num_synapses matrix containing spike counts for each synapse, for each time evaluation step
        self.delta_t = self.construct_dt_matrix()  # construct the elapsed time correlation for spike history matrix
        self.last_history_update_time = -1.0  # this is the time at which the history array was last updated

        # stdp parameters
        # self.num_stdp_histories = int(self.stdpp.stdp_window / self.tki.dt())
        # self.stdp_r1 = np.zeros((self.num_stdp_histories, self.num_synapses), dtype=np.float)
        # self.stdp_r2 = np.zeros((self.num_stdp_histories, self.num_synapses), dtype=np.float)
        # self.stdp_o1 = np.zeros((self.num_stdp_histories, self.num_synapses), dtype=np.float)
        # self.stdp_o2 = np.zeros((self.num_stdp_histories, self.num_synapses), dtype=np.float)
        self.stdp_r1 = np.zeros((self.m, self.n), dtype=np.float)
        self.stdp_r2 = np.zeros((self.m, self.n), dtype=np.float)
        self.stdp_o1 = np.zeros((self.m, self.n), dtype=np.float)
        self.stdp_o2 = np.zeros((self.m, self.n), dtype=np.float)

        self.last_stdp_r2 = np.zeros((self.m, self.n), dtype=np.float)
        self.last_stdp_o2 = np.zeros((self.m, self.n), dtype=np.float)

    def construct_weights(self):
        """
        This generates the weight matrix. This should be overriden for different connection types
        """
        if self.initial_w is None:
            w = np.random.uniform(low=self.w_rand_min, high=self.w_rand_max, size=(self.m, self.n))
        else:
            w = np.full((self.m, self.n), self.initial_w)
        
        return w

    def construct_dt_matrix(self):
        """
        Construct the matrix that relates the rows of self.history to the elapsed time. This should
        only be called once on initialization
        """
        delta_t = np.zeros(self.history.shape, dtype=float)
        times = np.arange(1, self.num_histories+1, 1) * self.tki.dt()
        for idx, val in np.ndenumerate(times):
            delta_t[idx, :] = val
    
        return delta_t

    def roll_history_and_assign(self, assignment):
        """
        Roll the spike history to timestamp t-1 and assign the latest incoming spikes
        """
        if self.tki.tick_time() == self.last_history_update_time:
            raise RuntimeError("An attempt was made to modify the synaptic history matrix more than once in a single time step.")
        
        self.history = fast_row_roll(self.history, assignment)  
        
        self.last_history_update_time = self.tki.tick_time()

    def pre_fire_notify(self, fired_neurons):
        """
        Notify this synaptic group of pre-synaptic neuron spikes. This is used for both updating the spike
        history and running pre-spike online STDP training
        """
        # we need to reshape this to the size of the weight matrix, so that we only update weights of synapses
        # that connect to this particular neuron, and also so that we don't store spike histories of neurons
        # that didn't spike
        a = np.zeros((self.m, self.n))
        # print(a)
        # print(fired_neurons)
        # print(np.where(fired_neurons>0.5))
        # x[0,np.where(y>0),:] = y[np.where(y>0),None]
        a[np.where(fired_neurons>0.5), :] = fired_neurons[np.where(fired_neurons>0.5), None]
        # print(a)
        
        self.roll_history_and_assign(a)
        if self.trainable:
            self._stdp(a, 'pre') 

    def post_fire_notify(self, fired_neurons):
        """
        Notify this synaptic group of post-synaptic neuron spikes. This is used for both updating the spike
        history and running post-spike online STDP training
        """
        # we need to reshape this to the size of the weight matrix, so that we only update weights of synapses
        # that connect to this particular neuron
        a = np.zeros((self.m, self.n))
        a[:, np.where(fired_neurons>0.5)] = fired_neurons[np.where(fired_neurons>0.5)]
        if self.trainable:
            self._stdp(a, 'pre')  

    def _stdp(self, fired_neurons, fire_time):
        """
        Runs online STDP algorithm: fired_neurons is just the spike count array
        """
        # if self.tki.tick_time() == self.last_history_update_time:
        #     raise RuntimeError("An attempt was made to run the STDP training process on a single synaptic group more than once in a single time step.")
        
        # reset what is considered the previous r2 and o2 parameters
        self.last_stdp_o2 = self.stdp_o2.copy()
        self.last_stdp_r2 = self.stdp_r2.copy()

        # calculate change in STDP spike trace parameters using Euler's method
        # dr1 = -1.0 * self.tki.dt() * self.stdp_r1[0] / self.stdpp.tao_plus
        # dr2 = -1.0 * self.tki.dt() * self.stdp_r2[0] / self.stdpp.tao_x
        # do1 = -1.0 * self.tki.dt() * self.stdp_o1[0] / self.stdpp.tao_minus
        # do2 = -1.0 * self.tki.dt() * self.stdp_o2[0] / self.stdpp.tao_y
        dr1 = -1.0 * self.tki.dt() * self.stdp_r1 / self.stdpp.tao_plus
        dr2 = -1.0 * self.tki.dt() * self.stdp_r2 / self.stdpp.tao_x
        do1 = -1.0 * self.tki.dt() * self.stdp_o1 / self.stdpp.tao_minus
        do2 = -1.0 * self.tki.dt() * self.stdp_o2 / self.stdpp.tao_y

        # roll and update the STDP spike traces based on the differentials just calculated
        # self.stdp_r1 = fast_row_roll(self.stdp_r1, dr1 + self.stdp_r1[0])
        # self.stdp_r2 = fast_row_roll(self.stdp_r2, dr2 + self.stdp_r2[0])
        # self.stdp_o1 = fast_row_roll(self.stdp_o1, do1 + self.stdp_o1[0])
        # self.stdp_o2 = fast_row_roll(self.stdp_o2, do2 + self.stdp_o2[0])
        self.stdp_r1 = dr1 + self.stdp_r1
        self.stdp_r2 = dr2 + self.stdp_r2
        self.stdp_o1 = do1 + self.stdp_o1
        self.stdp_o2 = do2 + self.stdp_o2

        # find the indices where there were spikes
        si = np.where(fired_neurons > 0)
        
        # calculate new weights and stdp parameters based on firing locations
        if fire_time == 'pre':
            self.stdp_r1[si] += 1.0
            self.stdp_r2[si] += 1.0
            self.w[si] -= self.stdp_o1[si] * (self.stdpp.a2_minus + self.stdpp.a3_minus * self.last_stdp_r2[si])
        elif fire_time == 'post':
            self.stdp_o1[si] += 1.0
            self.stdp_o2[si] += 1.0
            self.w[si] += self.stdp_r1[si] * (self.stdpp.a2_plus + self.stdpp.a3_plus * self.last_stdp_o2[si])

    def calc_isyn(self):
        """
        Calculate the current flowing across this synaptic group, as a function of the spike history
        """
        # print(self.history.shape)
        # print(self.w.shape)
        # print(self.post_n.v_m.shape)
        # print(self.pre_n.v_rev.shape)
        # print(self.pre_n.gbar.shape)
        # print(self.delta_t.shape)
        # exit()
        v_m_post = np.zeros((self.m, self.n), dtype=np.float)
        v_rev_pre = np.zeros((self.m, self.n), dtype=np.float)
        gbar_pre = np.zeros((self.m, self.n), dtype=np.float)

        v_m_post[:] = self.post_n.v_m
        v_rev_pre.T[:] = self.pre_n.v_rev
        gbar_pre.T[:] = self.pre_n.gbar
        # return np.sum(self.history * self.w * (self.post_n.v_m - self.pre_n.v_rev) * self.pre_n.gbar * np.exp(-1.0 * self.delta_t / self.synp.tao_syn))
        return np.sum(self.history * self.w * (v_m_post - v_rev_pre) * gbar_pre * np.exp(-1.0 * self.delta_t / self.synp.tao_syn))
    
    def reset(self):
        """
        Reset the synaptic parameters to initial conditions, except for the weight matrix
        """
        self.history.fill(0.0)


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

    def fully_connect(self, g1_tag, g2_tag, connection_probability=1.0, trainable=True, w_i=None, skip_self=False, learning_params=None, minw=0.5, maxw=0.9):
        """
        Connect all of the neurons in g1 to every neuron in g2 with a given probability
        if skip_self is set to True, then for each neuron, n1 in g1, connect n1 to each neuron, n2 in g2, 
        ONLY IF n2 DOES NOT have an axon projecting to n1. This is useful for lateral inhibition.
        """
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        s = SynapticGroup(g1, g2, self.tki)

        # store the new synaptic group into memory
        self.synapses.append(s)
    
    def get_w_between(self, g1_tag, g2_tag):
        """
        Returns the weight matrix between two neural groups
        """
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        for s in self.synapses:
            if s.pre_n == g1 and s.post_n == g2:
                return s.w 
        
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
                for s in self.synapses:
                    # if this synapse group is presynaptic, then calculate the current coming across it and run that current through the current neural group
                    if s.post_n == g:
                        i = s.calc_isyn()
                        g.run(i)
                        
                        # send out spikes to outgoing synapses
                        for s2 in self.synapses:
                            # if this neuron group is the presynaptic group to this synapse group
                            if s2.pre_n == g:
                                # roll the synaptic spike history back a step and assign the new spikes
                                # s2.roll_history_and_assign(g.spike_count)
                                s2.pre_fire_notify(g.spike_count)
                            else:
                                s2.post_fire_notify(g.spike_count)

        
        # if train:
        #     # update synaptic weights
        #     # loop over every synapse in the network
        #     for s in self.synapses:
        #         s.stdp()
    
    # def rest(self):
    #     for g in self.neural_groups:
    #         for n in g.n:
    #             n.v_membrane = n.reset()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys

    # mode = "single_neuron"
    mode = "single_network"
    # single neuron simulation
    if mode == "single_neuron":
        tki = TimeKeeperIterator(timeunit=0.1*msec)
        duration = 1000.0 * msec
        n = AdExNeuralGroup(np.ones((1,1), dtype=np.int), "George", tki, AdExParams())

        vms = []

        for step in tki:
            
            vms.append(n.run(-1 * namp)[0][0])
            
            # sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

            if step >= duration/tki.dt():
                break

        times = np.arange(0,len(vms), 1) * tki.dt() / msec
        plt.plot(times, vms)
        plt.title("Voltage Track")
        plt.xlabel("Time (msec)")
        plt.ylabel("Membrane Potential (mvolt)")
        plt.show()
    
    if mode == "single_network":
        import time
        start = time.time()
        tki = TimeKeeperIterator(timeunit=0.1*msec)
        duration = 1000.0 * msec
        g1 = SensoryNeuralGroup(np.ones(2, dtype=np.int), "Luny", tki, AdExParams())
        g2 = AdExNeuralGroup(np.ones(3, dtype=np.int), "George", tki, AdExParams())
        g3 = AdExNeuralGroup(np.ones(2, dtype=np.int), "Ada", tki, AdExParams())
        g3.tracked_vars = ["v_m"]

        nn = NeuralNetwork([g1, g2, g3], "blah", tki)
        nn.fully_connect("Luny", "George", w_i=1.0)
        nn.fully_connect("George", "Ada", w_i=1.0)
        # nn.set_trainability(False)

        vms = []

        for step in tki:
            # inject spikes into sensory layer
            g1.run(poisson_train(0.1*np.ones(g1.shape, dtype=np.float), tki.dt(), 64))
            # run all layers
            nn.run_order(["Luny", "George", "Ada"])
            
            # sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

            if step >= duration/tki.dt():
                break
        
        print(time.time()-start)
        times = np.arange(0,len(g3.v_m_track), 1) * tki.dt() / msec

        v = np.array(g3.v_m_track).T[0][0]

        
        plt.plot(times, v)
        plt.title("Voltage Track")
        plt.xlabel("Time (msec)")
        plt.ylabel("Membrane Potential (mvolt)")
        plt.show()

        print(nn.get_w_between("Luny", "George"))
