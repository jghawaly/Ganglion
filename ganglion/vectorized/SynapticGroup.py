from numba import jit, prange
import numpy as np
from NeuralGroup import NeuralGroup
from parameters import SynapseParams, STDPParams
import multiprocessing as mp


@jit(nopython=True, parallel=True, nogil=True)
def fast_row_roll(val, assignment):
    """
    A JIT compiled method for roll the values of all rows in a givenm matrix down by one, and
    assigning the first row to the given assignment
    """
    val[1:] = val[0:-1]
    val[0] = assignment
    return val


class SynapticGroup:
    """
    Defines a groups of synapses connecting two groups of neurons
    """
    def __init__(self, pre_n: NeuralGroup, post_n: NeuralGroup, tki, initial_w: float=None, w_rand_min: float=0.0, w_rand_max: float=1.0, trainable: bool=True, syn_params: SynapseParams=None, stdp_params: STDPParams=None, weight_multiplier=None, stdp_form='pair'):
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
        self.history = np.zeros((self.num_histories, self.m), dtype=np.float)  # A num_histories * num_synapses matrix containing spike counts for each synapse, for each time evaluation step
        self.last_history_update_time = -1.0  # this is the time at which the history array was last updated
        self.p_term = self.precalculate_term()  # precalculate part of the synaptic current calculation

        # triplet stdp parameters THESE ARE NOT LONGER SUPPORTED ? <- maybe?
        self.stdp_r1 = np.zeros((self.m, self.n), dtype=np.float)
        self.stdp_r2 = np.zeros((self.m, self.n), dtype=np.float)
        self.stdp_o1 = np.zeros((self.m, self.n), dtype=np.float)
        self.stdp_o2 = np.zeros((self.m, self.n), dtype=np.float)

        self.last_stdp_r2 = np.zeros((self.m, self.n), dtype=np.float)
        self.last_stdp_o2 = np.zeros((self.m, self.n), dtype=np.float)
        # ------------------------------------------------------------
        # pair stdp parameters
        self.stdp_pre = np.zeros((self.m, self.n), dtype=np.float)
        self.stdp_post = np.zeros((self.m, self.n), dtype=np.float)

        # use selected form of stdp
        if stdp_form == 'pair':
            self._stdp = self._pair_stdp
        elif stdp_form == "triplet":
            self._stdp = self._triplet_stdp
        else:
            raise RuntimeError("Invalid stdp form, must be pair or triplet")

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
    
    def precalculate_term(self):
        delta_t = np.zeros(self.num_histories, dtype=float)
        times = np.arange(1, self.num_histories+1, 1) * self.tki.dt()
        for idx, val in np.ndenumerate(times):
            delta_t[idx] = val

        alpha_time = delta_t / self.synp.tao_syn * np.exp(-1.0 * delta_t / self.synp.tao_syn)

        gbar_pre = np.zeros(self.m, dtype=np.float).transpose()
        gbar_pre[:] = self.pre_n.gbar

        return np.outer(gbar_pre.transpose(), alpha_time)

    def roll_history_and_assign(self, assignment):
        """
        Roll the spike history to timestamp t-1 and assign the latest incoming spikes
        """
        # if self.tki.tick_time() == self.last_history_update_time:
        #     print(self.pre_n.name)
        #     print(self.post_n.name)
        #     raise RuntimeError("An attempt was made to modify the synaptic history matrix more than once in a single time step.")
        
        self.history = fast_row_roll(self.history, assignment)  
        
        self.last_history_update_time = self.tki.tick_time()
    
    def pre_fire_notify(self, fired_neurons):
        """
        Notify this synaptic group of pre-synaptic neuron spikes. This is used for both updating the spike
        history and running pre-spike online STDP training
        """
        f = np.where(fired_neurons>0.5)

        # increment presynaptic trace (triplet stdp)
        self.stdp_r1[f, :] += 1
        self.stdp_r2[f, :] += 1

        # increment presynaptic trace (standard stdp)
        self.stdp_pre[f,:] += 1.0
        
        self.roll_history_and_assign(fired_neurons)
        if self.trainable:
            self._stdp(fired_neurons, 'pre') 

    def post_fire_notify(self, fired_neurons):
        """
        Notify this synaptic group of post-synaptic neuron spikes. This is used for both updating the spike
        history and running post-spike online STDP training
        """
        f = np.where(fired_neurons>0.5)
        
        # increment postsynaptic trace (triplet stdp)
        self.stdp_o1[:, f] += 1
        self.stdp_o2[:, f] += 1

        # increment postsynaptic trace (standard stdp)
        self.stdp_post[:,f] += 1.0

        if self.trainable:
            self._stdp(fired_neurons, 'post')  

    def _pair_stdp(self, fired_neurons, fire_time):
        """
        Runs online standard STDP algorithm with multiplicative weight dependence
        @param fired_neurons: The spike count array
        @param fire_time: 'pre' for presynaptic firing and 'post' for postsynaptic firing
        """
        # calculate change in STDP spike trace parameters using Euler's method
        self.stdp_pre += -1.0 * self.tki.dt() * self.stdp_pre / self.stdpp.stdp_tao_pre
        self.stdp_post += -1.0 * self.tki.dt() * self.stdp_post / self.stdpp.stdp_tao_post

        # find the indices where there were spikes
        si = np.where(fired_neurons > 0)
        
        # calculate new weights and stdp parameters based on firing locations
        if fire_time == 'pre':
            # # increment presynaptic trace
            # self.stdp_pre[si,:] += 1.0
            # apply weight change, as a function of the postsynaptic trace
            self.w[si,:] += self.stdpp.pre_multipler * self.stdpp.lr_pre * self.w[si,:] * self.stdp_post[si,:]
        elif fire_time == 'post':
            # # increment postsynaptic trace
            # self.stdp_post[:,si] += 1.0
            # apply weight change, as a function of the presynaptic trace
            self.w[:,si] += self.stdpp.post_multiplier * self.stdpp.lr_post * (1.0 - self.w[:,si]) * self.stdp_pre[:,si]

    def _triplet_stdp(self, fired_neurons, fire_time):
        """
        Runs online triplet STDP algorithm with hard weight bounds: NO LONGER SUPPORTED
        """
        # if self.tki.tick_time() == self.last_history_update_time:
        #     raise RuntimeError("An attempt was made to run the STDP training process on a single synaptic group more than once in a single time step.")
        
        # reset what is considered the previous r2 and o2 parameters
        self.last_stdp_o2 = self.stdp_o2.copy()
        self.last_stdp_r2 = self.stdp_r2.copy()

        # calculate change in STDP spike trace parameters using Euler's method
        self.stdp_r1 += -1.0 * self.tki.dt() * self.stdp_r1 / self.stdpp.tao_plus
        self.stdp_r2 += -1.0 * self.tki.dt() * self.stdp_r2 / self.stdpp.tao_x
        self.stdp_o1 += -1.0 * self.tki.dt() * self.stdp_o1 / self.stdpp.tao_minus
        self.stdp_o2 += -1.0 * self.tki.dt() * self.stdp_o2 / self.stdpp.tao_y

        # find the indices where there were spikes
        si = np.where(fired_neurons > 0)
        
        # calculate new weights and stdp parameters based on firing locations
        if fire_time == 'pre':
            # self.stdp_r1[si,:] += 1.0
            # self.stdp_r2[si,:] += 1.0
            dw = self.stdp_o1[si,:] * (self.stdpp.a2_minus + self.stdpp.a3_minus * self.last_stdp_r2[si,:])
            
            self.w[si,:] = np.clip(self.w[si,:] - self.stdpp.lr * dw, 0.0, 1.0)
        elif fire_time == 'post':
            # self.stdp_o1[:,si] += 1.0
            # self.stdp_o2[:,si] += 1.0

            dw = self.stdp_r1[:,si] * (self.stdpp.a2_plus + self.stdpp.a3_plus * self.last_stdp_o2[:,si])
            
            self.w[:,si] = np.clip(self.w[:,si] + self.stdpp.lr * dw, 0.0, 1.0)

    def calc_isyn(self):
        """
        Calculate the current flowing across this synaptic group, as a function of the spike history
        """
        v_m_post = np.zeros((self.m, self.n), dtype=np.float)
        v_rev_pre = np.zeros((self.m, self.n), dtype=np.float)

        v_m_post[:] = self.post_n.v_m
        v_rev_pre.T[:] = self.pre_n.v_rev
        v_term = v_rev_pre-v_m_post

        i_syn = np.zeros(self.n, dtype=np.float)
        for k in range(self.num_histories):
            hist_k = self.history[k]
            p_term_k = self.p_term[:, k][np.newaxis].transpose()
            i_current = np.dot(hist_k, self.w*p_term_k*v_term)
            i_syn += i_current
        
        return i_syn


if __name__ == "__main__":
    print()
    print("Running Unit Test...")
    print()
    from NeuralGroup import ExLIFNeuralGroup, SensoryNeuralGroup
    from NeuralNetwork import NeuralNetwork
    from timekeeper import TimeKeeperIterator
    from parameters import ExLIFParams
    from units import *
    import time


    tki = TimeKeeperIterator(timeunit=0.01*msec)
    duration = 5 * msec

    g1 = SensoryNeuralGroup(np.ones(4, dtype=np.int), "input", tki, ExLIFParams())
    g2 = ExLIFNeuralGroup(np.ones(4, dtype=np.int), "hidden", tki, ExLIFParams())
    g3 = ExLIFNeuralGroup(np.ones(4, dtype=np.int), "output", tki, ExLIFParams())
    g3.tracked_vars = ['i_syn']

    nn = NeuralNetwork([g1, g2, g3], "network", tki)
    
    nn.fully_connect("input", "hidden", trainable=False, w_i=0.1)
    nn.fully_connect("hidden", "output", trainable=False, w_i=0.1)
    
    start_time = time.time()
    for step in tki:
        g1.run(np.array([1,1,1,1]))

        nn.run_order(["input", "hidden", "output"])

        if step >= duration/tki.dt():
            break
    end_time = time.time()
    tsc = np.sum(g3.isyn_track)

    if 1.4e-6 > tsc > 1.3e-6:
        test = "PASSED"
    else:
        test = "FAILED"
    print("Unit Test  ::  %s  ::  Total synaptic current  ::  %g  ::  Expected synaptic current  ::  %s  ::  Execution Time  ::  %g seconds" % (test, np.sum(g3.isyn_track), "1.36319e-06", end_time-start_time))
        
