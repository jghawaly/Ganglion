from numba import jit, prange
import numpy as np
from NeuralGroup import AdExNeuralGroup
from parameters import SynapseParams, STDPParams
import multiprocessing as mp


@jit(nopython=True, parallel=True, nogil=True)
def fast_row_roll(val, assignment):
    """
    A JIT compiled method for roll the values of all rows in a givenm matrix down by one, and
    assigning the first row to the given assignment
    """
    val[1:,:] = val[0:-1,:]
    val[0] = assignment
    return val


@jit(nopython=True)#, parallel=True, nogil=True)
def isyn_jit_old(history, w, v_m_post, v_rev_pre, gbar_pre, delta_t, tao_syn):
    return np.sum(history * w * (v_m_post - v_rev_pre) * gbar_pre * np.exp(-1.0 * delta_t / tao_syn))

@jit(nopython=True)
def isyn_jit(history, w, v_m_post, v_rev_pre, gbar_pre, decayed_time):
    """
    Calculate synaptic current, JIT compiled for faster processing
    """
    return np.sum(history * w * (v_m_post - v_rev_pre) * gbar_pre * decayed_time)

@jit(nopython=True, parallel=True, nogil=True, fastmath=True)
def isyn_jit_parallel(history, w, v_m_post, v_rev_pre, gbar_pre, decayed_time, weight_multiplier):
    """
    Calculate synaptic current, JIT compiled for faster processing
    """
    return history * w * (v_m_post - v_rev_pre) * gbar_pre * decayed_time * weight_multiplier

@jit(nopython=True, parallel=True, nogil=True)
def isyn_jit_full_parallel(history, w, v_m_post, v_rev_pre, gbar_pre, decayed_time, weight_multiplier):
    """
    Run the isyn calculation for all timesteps in the history array and return the resulting array
    """
    res = np.zeros_like(history)
    for i in prange(history.shape[0]):
        res[i] = isyn_jit_parallel(history[i], w, v_m_post, v_rev_pre, gbar_pre, decayed_time[i], weight_multiplier)
    return res

class SynapticGroup:
    """
    Defines a groups of synapses connecting two groups of neurons
    """
    def __init__(self, pre_n: AdExNeuralGroup, post_n: AdExNeuralGroup, tki, initial_w: float=None, w_rand_min: float=0.0, w_rand_max: float=1.0, trainable: bool=True, syn_params: SynapseParams=None, stdp_params: STDPParams=None, weight_multiplier=None):
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
        if weight_multiplier is None:
            self.weight_multiplier = np.ones((self.m, self.n), dtype=np.float)
        else:
            self.weight_multiplier = weight_multiplier
        self.w = self.construct_weights()  # construct weight matrix
        
        self.pre_spikes = []
        self.post_spikes = []

        self.num_histories = int(self.synp.spike_window / self.tki.dt())  # number of discretized time bins that we will keep track of presynaptic spikes in
        self.history = np.zeros((self.num_histories, self.m, self.n), dtype=np.float)  # A num_histories * num_synapses matrix containing spike counts for each synapse, for each time evaluation step
        self.delta_t = self.construct_dt_matrix()  # construct the elapsed time correlation for spike history matrix
        self.last_history_update_time = -1.0  # this is the time at which the history array was last updated
        self.time_decay_matrix = np.exp(-1.0 * self.delta_t / self.synp.tao_syn)  # this is the time decay matrix that is used for decaying the PSPs, precalculated for faster processing

        # stdp parameters
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
        if self.weight_multiplier.shape != (self.m, self.n):
            raise ValueError("The shape of the weight multiplier matrix must be the same shape of the weight matrix but are : %s and %s" % (str(self.weight_multiplier.shape), str((self.m, self.n))))

        if self.initial_w is None:
            w = np.random.uniform(low=self.w_rand_min, high=self.w_rand_max, size=(self.m, self.n))
        else:
            w = np.full((self.m, self.n), self.initial_w)
        
        return w * self.weight_multiplier

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
            print(self.pre_n.name)
            print(self.post_n.name)
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
        f = np.where(fired_neurons>0.5)
        a = np.zeros((self.m, self.n))
        a[f, :] = fired_neurons[f, None]

        # we don't want to update synapses with zero weight
        mask = self.w.copy()
        mask[np.where(self.w > 0.0)] = 1.0

        self.stdp_r1[f, :] += 1
        self.stdp_r2[f, :] += 1
        
        self.roll_history_and_assign(a)
        if self.trainable:
            self._stdp(a * mask, 'pre') 

    def post_fire_notify(self, fired_neurons):
        """
        Notify this synaptic group of post-synaptic neuron spikes. This is used for both updating the spike
        history and running post-spike online STDP training
        """
        # we need to reshape this to the size of the weight matrix, so that we only update weights of synapses
        # that connect to this particular neuron
        f = np.where(fired_neurons>0.5)
        a = np.zeros((self.m, self.n))
        a[:, f] = fired_neurons[f]

        # we don't want to update synapses with zero weight
        mask = self.w.copy()
        mask[np.where(self.w > 0.0)] = 1.0
        
        self.stdp_o1[:, f] += 1
        self.stdp_o2[:, f] += 1

        if self.trainable:
            self._stdp(a * mask, 'post')  

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
        dr1 = -1.0 * self.tki.dt() * self.stdp_r1 / self.stdpp.tao_plus
        dr2 = -1.0 * self.tki.dt() * self.stdp_r2 / self.stdpp.tao_x
        do1 = -1.0 * self.tki.dt() * self.stdp_o1 / self.stdpp.tao_minus
        do2 = -1.0 * self.tki.dt() * self.stdp_o2 / self.stdpp.tao_y

        # roll and update the STDP spike traces based on the differentials just calculated
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

            dw = self.stdp_o1[si] * (self.stdpp.a2_minus + self.stdpp.a3_minus * self.last_stdp_r2[si])
            
            self.w[si] = np.clip(self.w[si] - self.stdpp.lr * dw, 0.0, 1.0)
        elif fire_time == 'post':
            self.stdp_o1[si] += 1.0
            self.stdp_o2[si] += 1.0

            dw = self.stdp_r1[si] * (self.stdpp.a2_plus + self.stdpp.a3_plus * self.last_stdp_o2[si])
            
            self.w[si] = np.clip(self.w[si] + self.stdpp.lr * dw, 0.0, 1.0)

    def calc_isyn(self):
        """
        Calculate the current flowing across this synaptic group, as a function of the spike history
        """
        v_m_post = np.zeros((self.m, self.n), dtype=np.float)
        v_rev_pre = np.zeros((self.m, self.n), dtype=np.float)
        gbar_pre = np.zeros((self.m, self.n), dtype=np.float)

        v_m_post[:] = self.post_n.v_m
        v_rev_pre.T[:] = self.pre_n.v_rev
        gbar_pre.T[:] = self.pre_n.gbar

        # print(self.history.shape)
        # print(self.time_decay_matrix.shape)
        # print(self.history[0])
        # exit()
    
        # return np.sum(self.history * self.w * (v_m_post - v_rev_pre) * gbar_pre * np.exp(-1.0 * self.delta_t / self.synp.tao_syn))  # this one is slow and works
        # return isyn_jit_old(self.history, self.w, v_m_post, v_rev_pre, gbar_pre, self.delta_t, self.synp.tao_syn) # this one is faster and works
        # return isyn_jit(self.history, self.w, v_m_post, v_rev_pre, gbar_pre, self.time_decay_matrix) # this one is even faster and works

        # this is the fastest, and works
        return np.sum(isyn_jit_full_parallel(self.history, self.w, v_m_post, v_rev_pre, gbar_pre, self.time_decay_matrix, self.weight_multiplier))

        
