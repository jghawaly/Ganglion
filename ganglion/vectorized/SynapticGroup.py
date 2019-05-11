from numba import jit, prange
import numpy as np
from NeuralGroup import NeuralGroup
from parameters import SynapseParams, PairSTDPParams, TripletSTDPParams, DASTDPParams
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


# class SynapticGroup:
#     """
#     Defines a groups of synapses connecting two groups of neurons
#     """
#     def __init__(self, pre_n: NeuralGroup, post_n: NeuralGroup, tki, initial_w: float=None, w_rand_min: float=0.0, w_rand_max: float=1.0, 
#                  trainable: bool=True, syn_params: SynapseParams=None, stdp_params: STDPParams=None, 
#                  weight_multiplier=None, stdp_form='pair', loaded_weights=None):
#         self.synp = SynapseParams() if syn_params is None else syn_params  # synapse parameters are default if None is given
#         self.stdpp = STDPParams() if stdp_params is None else stdp_params  # STDP parameters are default if None is given
#         self.tki = tki  # reference to timekeeper object that is shared amongst the entire network
#         self.pre_n = pre_n  # presynaptic neural group
#         self.post_n = post_n  # postsynaptic neural group
#         self.trainable = trainable  # True if this synaptic group is trainable, False otherwise

#         self.m = pre_n.shape[0]  # number of neurons in presynaptic group
#         self.n = post_n.shape[0]  # number of neuronsin postsynaptic group
#         self.num_synapses = self.m * self.n  # nunmber of synapses to be generated
#         self.w_rand_min = w_rand_min  # minimum weight for weight-initialization via uniform distribution
#         self.w_rand_max = w_rand_max  # maximum weight for weight-initialization via uniform distribution
#         self.initial_w = initial_w  # initial weight value for static weight initialization
#         if weight_multiplier is None:
#             self.weight_multiplier = np.ones((self.m, self.n), dtype=np.float)
#         else:
#             self.weight_multiplier = weight_multiplier
#         self.w = self.construct_weights(loaded_weights)  # construct weight matrix
        
#         self.pre_spikes = []
#         self.post_spikes = []

#         self.num_histories = int(self.synp.spike_window / self.tki.dt())  # number of discretized time bins that we will keep track of presynaptic spikes in
#         self.history = np.zeros((self.num_histories, self.m), dtype=np.float)  # A num_histories * num_synapses matrix containing spike counts for each synapse, for each time evaluation step
#         self.last_history_update_time = -1.0  # this is the time at which the history array was last updated
#         self.p_term = self.precalculate_term()  # precalculate part of the synaptic current calculation

#         # triplet stdp parameters 
#         self.stdp_r1 = np.zeros((self.m, self.n), dtype=np.float)
#         self.stdp_r2 = np.zeros((self.m, self.n), dtype=np.float)
#         self.stdp_o1 = np.zeros((self.m, self.n), dtype=np.float)
#         self.stdp_o2 = np.zeros((self.m, self.n), dtype=np.float)

#         # ------------------------------------------------------------
#         # pair stdp parameters
#         self.stdp_pre = np.zeros((self.m, self.n), dtype=np.float)
#         self.stdp_post = np.zeros((self.m, self.n), dtype=np.float)

#         # use selected form of stdp
#         if stdp_form == 'pair':
#             self._stdp = self._pair_stdp
#         elif stdp_form == "triplet":
#             self._stdp = self._triplet_stdp
#         else:
#             raise RuntimeError("Invalid stdp form, must be pair or triplet")

#     def reset(self):
#         self.history.fill(0)
#         self.stdp_r1.fill(0)
#         self.stdp_r2.fill(0)
#         self.stdp_o1.fill(0)
#         self.stdp_o2.fill(0)

#         self.stdp_pre.fill(0)
#         self.stdp_post.fill(0)

#     def construct_weights(self, loaded_weights):
#         """
#         This generates the weight matrix. This should be overriden for different connection types
#         """
#         if self.weight_multiplier.shape != (self.m, self.n):
#             raise ValueError("The shape of the weight multiplier matrix must be the same shape of the weight matrix but are : %s and %s" % (str(self.weight_multiplier.shape), str((self.m, self.n))))

#         # check if pre-loaded weights were given
#         if loaded_weights is not None:
#             w = loaded_weights
#         else:
#             # create the weights
#             if self.initial_w is None:
#                 w = np.random.uniform(low=self.w_rand_min, high=self.w_rand_max, size=(self.m, self.n))
#             else:
#                 w = np.full((self.m, self.n), self.initial_w)
        
#         return w * self.weight_multiplier

#     def save_weights(self, path):
#         np.save(path, self.w)

#     def construct_dt_matrix(self):
#         """
#         Construct the matrix that relates the rows of self.history to the elapsed time. This should
#         only be called once on initialization
#         """
#         delta_t = np.zeros(self.history.shape, dtype=float)
#         times = np.arange(1, self.num_histories+1, 1) * self.tki.dt()
#         for idx, val in np.ndenumerate(times):
#             delta_t[idx, :] = val

#         return delta_t
    
#     def precalculate_term(self):
#         delta_t = np.zeros(self.num_histories, dtype=float)
#         times = np.arange(1, self.num_histories+1, 1) * self.tki.dt()
#         for idx, val in np.ndenumerate(times):
#             delta_t[idx] = val

#         alpha_time = delta_t / self.synp.tao_syn * np.exp(-1.0 * delta_t / self.synp.tao_syn)

#         gbar_pre = np.zeros(self.m, dtype=np.float).transpose()
#         gbar_pre[:] = self.pre_n.gbar

#         return np.outer(gbar_pre.transpose(), alpha_time)

#     def roll_history_and_assign(self, assignment):
#         """
#         Roll the spike history to timestamp t-1 and assign the latest incoming spikes
#         """
#         if self.tki.tick_time() == self.last_history_update_time:
#             # if this history term has already been updated at this time step, then add to it rather than roll and reassign
#             self.history[0] += assignment
        
#         self.history = fast_row_roll(self.history, assignment)  
        
#         self.last_history_update_time = self.tki.tick_time()
    
#     def pre_fire_notify(self, fired_neurons):
#         """
#         Notify this synaptic group of pre-synaptic neuron spikes. This is used for both updating the spike
#         history and running pre-spike online STDP training
#         """
#         f = np.where(fired_neurons>0.5)

#         # increment triplet stdp standadr presynaptic trace
#         self.stdp_r1[f, :] += 1

#         # increment presynaptic trace (standard stdp)
#         self.stdp_pre[f,:] += 1.0
        
#         self.roll_history_and_assign(fired_neurons)
#         if self.trainable:
#             self._stdp(fired_neurons, 'pre') 
        
#         # icnrement triplet stdp tripley presynaptic trace
#         self.stdp_r2[f, :] += 1

#     def post_fire_notify(self, fired_neurons):
#         """
#         Notify this synaptic group of post-synaptic neuron spikes. This is used for both updating the spike
#         history and running post-spike online STDP training
#         """
#         f = np.where(fired_neurons>0.5)
        
#         # increment triplet stdp standard postsynaptic trace
#         self.stdp_o1[:, f] += 1

#         # increment postsynaptic trace (standard stdp)
#         self.stdp_post[:,f] += 1.0

#         if self.trainable:
#             self._stdp(fired_neurons, 'post')  
        
#         # increment triplet stdp triplet postsynaptic trace
#         self.stdp_o2[:, f] += 1

#     def _pair_stdp(self, fired_neurons, fire_time):
#         """
#         Runs online standard STDP algorithm with multiplicative weight dependence
#         @param fired_neurons: The spike count array
#         @param fire_time: 'pre' for presynaptic firing and 'post' for postsynaptic firing
#         """
#         # calculate change in STDP spike trace parameters using Euler's method
#         self.stdp_pre += -1.0 * self.tki.dt() * self.stdp_pre / self.stdpp.stdp_tao_pre
#         self.stdp_post += -1.0 * self.tki.dt() * self.stdp_post / self.stdpp.stdp_tao_post

#         # find the indices where there were spikes
#         si = np.where(fired_neurons > 0)
        
#         # calculate new weights and stdp parameters based on firing locations
#         if fire_time == 'pre':
#             # apply weight change, as a function of the postsynaptic trace
#             self.w[si,:] += self.stdpp.pre_multipler * self.stdpp.lr_pre * self.w[si,:] * self.stdp_post[si,:]
#         elif fire_time == 'post':
#             # apply weight change, as a function of the presynaptic trace
#             self.w[:,si] += self.stdpp.post_multiplier * self.stdpp.lr_post * (1.0 - self.w[:,si]) * self.stdp_pre[:,si]

#     def _triplet_stdp(self, fired_neurons, fire_time):
#         """
#         Runs online triplet STDP algorithm with hard weight bounds: NO LONGER SUPPORTED
#         """
#         # if self.tki.tick_time() == self.last_history_update_time:
#         #     raise RuntimeError("An attempt was made to run the STDP training process on a single synaptic group more than once in a single time step.")

#         # calculate change in STDP spike trace parameters using Euler's method
#         self.stdp_r1 += -1.0 * self.tki.dt() * self.stdp_r1 / self.stdpp.tao_plus
#         self.stdp_r2 += -1.0 * self.tki.dt() * self.stdp_r2 / self.stdpp.tao_x
#         self.stdp_o1 += -1.0 * self.tki.dt() * self.stdp_o1 / self.stdpp.tao_minus
#         self.stdp_o2 += -1.0 * self.tki.dt() * self.stdp_o2 / self.stdpp.tao_y

#         # find the indices where there were spikes
#         si = np.where(fired_neurons > 0)
        
#         # calculate new weights and stdp parameters based on firing locations
#         if fire_time == 'pre':
#             dw = self.stdp_o1[si,:] * (self.stdpp.a2_minus + self.stdpp.a3_minus * self.stdp_r2[si,:])
            
#             self.w[si,:] = np.clip(self.w[si,:] - self.stdpp.lr * dw, 0.0, 1.0)
#         elif fire_time == 'post':
#             dw = self.stdp_r1[:,si] * (self.stdpp.a2_plus + self.stdpp.a3_plus * self.stdp_o2[:,si])
            
#             self.w[:,si] = np.clip(self.w[:,si] + self.stdpp.lr * dw, 0.0, 1.0)

#     def calc_isyn(self):
#         """
#         Calculate the current flowing across this synaptic group, as a function of the spike history
#         """
#         v_m_post = np.zeros((self.m, self.n), dtype=np.float)
#         v_rev_pre = np.zeros((self.m, self.n), dtype=np.float)

#         v_m_post[:] = self.post_n.v_m
#         v_rev_pre.T[:] = self.pre_n.v_rev
#         v_term = v_rev_pre-v_m_post

#         i_syn = np.zeros(self.n, dtype=np.float)
#         for k in range(self.num_histories):
#             hist_k = self.history[k]
#             p_term_k = self.p_term[:, k][np.newaxis].transpose()
#             i_current = np.dot(hist_k, self.w*p_term_k*v_term)
#             i_syn += i_current
        
#         return i_syn
    
#     def scale_weights(self):
#         for i in range(self.w.shape[1]):
#             self.w[:,i] = self.w[:,i] / self.w[:,i].sum()

class BaseSynapticGroup:
    """
    Defines a groups of synapses connecting two groups of neurons
    """
    def __init__(self, pre_n: NeuralGroup, post_n: NeuralGroup, tki, initial_w: float=None, w_rand_min: float=0.0, w_rand_max: float=1.0, syn_params: SynapseParams=None, weight_multiplier=None, loaded_weights=None, trainable=True):
        self.synp = SynapseParams() if syn_params is None else syn_params  # synapse parameters are default if None is given
        self.tki = tki  # reference to timekeeper object that is shared amongst the entire network
        self.pre_n = pre_n  # presynaptic neural group
        self.post_n = post_n  # postsynaptic neural group
        self.trainable = trainable  # True if this is a trainable group 

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
        self.w = self.construct_weights(loaded_weights)  # construct weight matrix
        
        self.pre_spikes = []
        self.post_spikes = []

        self.num_histories = int(self.synp.spike_window / self.tki.dt())  # number of discretized time bins that we will keep track of presynaptic spikes in
        self.history = np.zeros((self.num_histories, self.m), dtype=np.float)  # A num_histories * num_synapses matrix containing spike counts for each synapse, for each time evaluation step
        self.last_history_update_time = -1.0  # this is the time at which the history array was last updated
        self.p_term = self.precalculate_term()  # precalculate part of the synaptic current calculation

    """ Methods that should NOT be overridden -----------------------------------------------------------------------------"""

    def construct_weights(self, loaded_weights):
        """
        This generates the weight matrix. This should be overriden for different connection types

        This should not be overridden by sub-classes
        """
        if self.weight_multiplier.shape != (self.m, self.n):
            raise ValueError("The shape of the weight multiplier matrix must be the same shape of the weight matrix but are : %s and %s" % (str(self.weight_multiplier.shape), str((self.m, self.n))))

        # check if pre-loaded weights were given
        if loaded_weights is not None:
            w = loaded_weights
        else:
            # create the weights
            if self.initial_w is None:
                w = np.random.uniform(low=self.w_rand_min, high=self.w_rand_max, size=(self.m, self.n))
            else:
                w = np.full((self.m, self.n), self.initial_w)
        
        return w * self.weight_multiplier
    
    def save_weights(self, path):
        """
        This saves the weight matrix to the designated path
        """
        np.save(path, self.w)
    
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
        if self.tki.tick_time() == self.last_history_update_time:
            # if this history term has already been updated at this time step, then add to it rather than roll and reassign
            self.history[0] += assignment
        
        self.history = fast_row_roll(self.history, assignment)  
        
        self.last_history_update_time = self.tki.tick_time()    
    
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
    
    def normalize_weights(self):
        """
        Normalize the weights such that they sum to one
        """
        for i in range(self.w.shape[1]):
            self.w[:,i] = self.w[:,i] / self.w[:,i].sum()

    """ Methods that can be overridden ---------------------------------------------------------------------------------"""

    def reset(self):
        """
        This is for resting the time-based parameters of the synaptic group.
        """
        self.history.fill(0)

    def pre_fire_notify(self, fired_neurons):
        """
        Notify this synaptic group of pre-synaptic neuron spikes. This is used for both updating the spike
        history and running online training
        """
        self.roll_history_and_assign(fired_neurons)

    def post_fire_notify(self, fired_neurons):
        """
        Notify this synaptic group of post-synaptic neuron spikes. This is used for both updating the spike
        history and runningonline training
        """
        pass


class PairSTDPSynapticGroup(BaseSynapticGroup):
    def __init__(self, pre_n: NeuralGroup, post_n: NeuralGroup, tki, initial_w: float=None, w_rand_min: float=0.0, w_rand_max: float=1.0, trainable: bool=True, syn_params: SynapseParams=None, weight_multiplier=None, loaded_weights=None, stdp_params: PairSTDPParams=None):
        super().__init__(pre_n, post_n, tki, initial_w=initial_w, w_rand_min=w_rand_min, w_rand_max=w_rand_max, syn_params=syn_params, weight_multiplier=weight_multiplier, loaded_weights=loaded_weights, trainable=trainable)
        
        # STDP parameters are default if None is given
        self.stdpp = PairSTDPParams() if stdp_params is None else stdp_params

        # pair STDP parameters
        self.stdp_pre = np.zeros((self.m, self.n), dtype=np.float)
        self.stdp_post = np.zeros((self.m, self.n), dtype=np.float)
    
    def reset(self):
        """
        This is for resting the time-based parameters of the synaptic group.
        """
        self.history.fill(0)
        self.stdp_pre.fill(0)
        self.stdp_post.fill(0)

    def pre_fire_notify(self, fired_neurons):
        """
        Notify this synaptic group of pre-synaptic neuron spikes. This is used for both updating the spike
        history and running pre-spike online STDP training
        """
        f = np.where(fired_neurons>0.5)

        # increment presynaptic trace (standard stdp)
        self.stdp_pre[f,:] += 1.0
        
        self.roll_history_and_assign(fired_neurons)
        if self.trainable:
            self.pair_stdp(fired_neurons, 'pre') 

    def post_fire_notify(self, fired_neurons):
        """
        Notify this synaptic group of post-synaptic neuron spikes. This is used for both updating the spike
        history and running post-spike online STDP training
        """
        f = np.where(fired_neurons>0.5)

        # increment postsynaptic trace (standard stdp)
        self.stdp_post[:,f] += 1.0

        if self.trainable:
            self.pair_stdp(fired_neurons, 'post')  

    def pair_stdp(self, fired_neurons, fire_time):
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
            # apply weight change, as a function of the postsynaptic trace
            self.w[si,:] += self.stdpp.pre_multipler * self.stdpp.lr_pre * self.w[si,:] * self.stdp_post[si,:]
        elif fire_time == 'post':
            # apply weight change, as a function of the presynaptic trace
            self.w[:,si] += self.stdpp.post_multiplier * self.stdpp.lr_post * (1.0 - self.w[:,si]) * self.stdp_pre[:,si]


class TripletSTDPSynapticGroup(BaseSynapticGroup):
    def __init__(self, pre_n: NeuralGroup, post_n: NeuralGroup, tki, initial_w: float=None, w_rand_min: float=0.0, w_rand_max: float=1.0, trainable: bool=True, syn_params: SynapseParams=None, weight_multiplier=None, loaded_weights=None, stdp_params: TripletSTDPParams=None):
        super().__init__(pre_n, post_n, tki, initial_w=initial_w, w_rand_min=w_rand_min, w_rand_max=w_rand_max, syn_params=syn_params, weight_multiplier=weight_multiplier, loaded_weights=loaded_weights, trainable=trainable)
        
        # STDP parameters are default if None is given
        self.stdpp = TripletSTDPParams() if stdp_params is None else stdp_params 

        # triplet STDP parameters 
        self.stdp_r1 = np.zeros((self.m, self.n), dtype=np.float)
        self.stdp_r2 = np.zeros((self.m, self.n), dtype=np.float)
        self.stdp_o1 = np.zeros((self.m, self.n), dtype=np.float)
        self.stdp_o2 = np.zeros((self.m, self.n), dtype=np.float)
    
    def reset(self):
        """
        This is for resting the time-based parameters of the synaptic group.
        """
        self.history.fill(0)
        self.stdp_r1.fill(0)
        self.stdp_r2.fill(0)
        self.stdp_o1.fill(0)
        self.stdp_o2.fill(0)

    def pre_fire_notify(self, fired_neurons):
        """
        Notify this synaptic group of pre-synaptic neuron spikes. This is used for both updating the spike
        history and running pre-spike online STDP training
        """
        f = np.where(fired_neurons>0.5)

        # increment triplet stdp standard presynaptic trace
        self.stdp_r1[f, :] += 1
        
        self.roll_history_and_assign(fired_neurons)
        if self.trainable:
            self.triplet_stdp(fired_neurons, 'pre') 
        
        # icnrement triplet stdp tripley presynaptic trace
        self.stdp_r2[f, :] += 1

    def post_fire_notify(self, fired_neurons):
        """
        Notify this synaptic group of post-synaptic neuron spikes. This is used for both updating the spike
        history and running post-spike online STDP training
        """
        f = np.where(fired_neurons>0.5)
        
        # increment triplet stdp standard postsynaptic trace
        self.stdp_o1[:, f] += 1

        if self.trainable:
            self.triplet_stdp(fired_neurons, 'post')  
        
        # increment triplet stdp triplet postsynaptic trace
        self.stdp_o2[:, f] += 1
    
    def triplet_stdp(self, fired_neurons, fire_time):
        """
        Runs online triplet STDP algorithm with hard weight bounds: NO LONGER SUPPORTED
        """
        # if self.tki.tick_time() == self.last_history_update_time:
        #     raise RuntimeError("An attempt was made to run the STDP training process on a single synaptic group more than once in a single time step.")

        # calculate change in STDP spike trace parameters using Euler's method
        self.stdp_r1 += -1.0 * self.tki.dt() * self.stdp_r1 / self.stdpp.tao_plus
        self.stdp_r2 += -1.0 * self.tki.dt() * self.stdp_r2 / self.stdpp.tao_x
        self.stdp_o1 += -1.0 * self.tki.dt() * self.stdp_o1 / self.stdpp.tao_minus
        self.stdp_o2 += -1.0 * self.tki.dt() * self.stdp_o2 / self.stdpp.tao_y

        # find the indices where there were spikes
        si = np.where(fired_neurons > 0)
        
        # calculate new weights and stdp parameters based on firing locations
        if fire_time == 'pre':
            dw = self.stdp_o1[si,:] * (self.stdpp.a2_minus + self.stdpp.a3_minus * self.stdp_r2[si,:])
            print(dw)
            
            self.w[si,:] = np.clip(self.w[si,:] - self.stdpp.lr * dw, 0.0, 1.0)
        elif fire_time == 'post':
            dw = self.stdp_r1[:,si] * (self.stdpp.a2_plus + self.stdpp.a3_plus * self.stdp_o2[:,si])
            print(dw)
            
            self.w[:,si] = np.clip(self.w[:,si] + self.stdpp.lr * dw, 0.0, 1.0)


class DASTDPSynapticGroup(BaseSynapticGroup):
    def __init__(self, pre_n: NeuralGroup, post_n: NeuralGroup, tki, initial_w: float=None, w_rand_min: float=0.0, w_rand_max: float=1.0, trainable: bool=True, syn_params: SynapseParams=None, weight_multiplier=None, loaded_weights=None, stdp_params: DASTDPParams=None):
        super().__init__(pre_n, post_n, tki, initial_w=initial_w, w_rand_min=w_rand_min, w_rand_max=w_rand_max, syn_params=syn_params, weight_multiplier=weight_multiplier, loaded_weights=loaded_weights, trainable=trainable)

        # STDP parameters are default if None is given
        self.stdpp = DASTDPParams() if stdp_params is None else stdp_params 

        # eligibility traces for LTP and LTD
        self.ltp_trace = np.zeros(self.w.shape, dtype=np.float)
        self.ltd_trace = np.zeros(self.w.shape, dtype=np.float)

        # pair STDP parameters
        self.stdp_pre = np.zeros((self.m, self.n), dtype=np.float)
        self.stdp_post = np.zeros((self.m, self.n), dtype=np.float)
    
    def reset(self):
        """
        This is for resting the time-based parameters of the synaptic group.
        """
        self.history.fill(0)
        self.stdp_pre.fill(0)
        self.stdp_post.fill(0)
        self.ltd_trace.fill(0)
        self.ltp_trace.fill(0)

    def pre_fire_notify(self, fired_neurons):
        """
        Notify this synaptic group of pre-synaptic neuron spikes. This is used for both updating the spike
        history and running pre-spike online STDP training
        """
        f = np.where(fired_neurons>0.5)

        # increment presynaptic trace (standard stdp)
        self.stdp_pre[f,:] += 1.0
        
        self.roll_history_and_assign(fired_neurons)
        
        if self.trainable:
            self.update_eligibility(fired_neurons, 'pre') 

    def post_fire_notify(self, fired_neurons):
        """
        Notify this synaptic group of post-synaptic neuron spikes. This is used for both updating the spike
        history and running post-spike online STDP training
        """
        f = np.where(fired_neurons>0.5)

        # increment postsynaptic trace (standard stdp)
        self.stdp_post[:,f] += 1.0

        if self.trainable:
            self.update_eligibility(fired_neurons, 'post')  

    def update_eligibility(self, fired_neurons, fire_time):
        """
        Updates the elibiility trace
        @param fired_neurons: The spike count array
        @param fire_time: 'pre' for presynaptic firing and 'post' for postsynaptic firing
        """
        # calculate change in STDP spike trace parameters using Euler's method
        self.stdp_pre += -1.0 * self.tki.dt() * self.stdp_pre / self.stdpp.stdp_tao_pre
        self.stdp_post += -1.0 * self.tki.dt() * self.stdp_post / self.stdpp.stdp_tao_post

        # calculate change in eligibility traces using Euler's method
        self.ltd_trace += -1.0 * self.tki.dt() * self.ltd_trace / self.stdpp.tao_ltd
        self.ltp_trace += -1.0 * self.tki.dt() * self.ltp_trace / self.stdpp.tao_ltp

        # find the indices where there were spikes
        si = np.where(fired_neurons > 0)
        
        # calculate new weights and stdp parameters based on firing locations
        if fire_time == 'pre':
            # increment LTD eligibility trace
            self.ltd_trace[si,:] += self.stdp_post[si,:]
        elif fire_time == 'post':
            # increment LTP eligibility trace
            self.ltp_trace[:,si] += self.stdp_pre[:,si]

    def apply_dopamine(self, reward):
        """
        Apply weight changes based on eligibility traces and reward

        Note: If reward is negative, then STDP will become inversed
        """
        # perform LTP based on reward
        self.w += self.ltp_trace * self.stdpp.lr_post * self.stdpp.post_multiplier * reward
        # perform LTD based on reward
        self.w += self.ltd_trace * self.stdpp.lr_pre * self.stdpp.pre_multipler * reward
        # make sure that weights stay in bounds
        self.w = np.clip(self.w, 0.0, 1.0)

        # reset eligibility traces
        self.ltd_trace.fill(0)
        self.ltp_trace.fill(0)
