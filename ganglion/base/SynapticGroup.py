from numba import jit, prange
import numpy as np
from NeuralGroup import NeuralGroup
from parameters import SynapseParams, PairSTDPParams, TripletSTDPParams, DASTDPParams
import multiprocessing as mp
import sys
np.set_printoptions(threshold=sys.maxsize)


@jit(nopython=True)
def fast_row_roll(val, assignment):
    """
    A JIT compiled method for roll the values of all rows in a given matrix down by one, and
    assigning the first row to the given assignment
    """
    val[1:] = val[0:-1]
    val[0] = assignment
    return val


@jit(nopython=True)
def any_nonzero(a):
    """
    A JIT compiled method for checking if any values in a matrix are nonzero
    """
    for v in a:
        if v != 0:
            return True
    return False


class BaseSynapticGroup:
    """
    Defines a groups of synapses connecting two groups of neurons
    """
    def __init__(self, pre_n: NeuralGroup, post_n: NeuralGroup, tki, initial_w: float=None, w_rand_min: float=0.0, w_rand_max: float=1.0, syn_params: SynapseParams=None, 
                 weight_multiplier=None, loaded_weights=None, trainable=True, localized_normalization=False, share_weights=False, enable_wd=False):
        self.synp = SynapseParams() if syn_params is None else syn_params  # synapse parameters are default if None is given
        self.tki = tki  # reference to timekeeper object that is shared amongst the entire network
        self.pre_n = pre_n  # presynaptic neural group
        self.post_n = post_n  # postsynaptic neural group
        self.trainable = trainable  # True if this is a trainable group 
        self.lati = self.synp.lati

        self.m = pre_n.shape[0]  # number of neurons in presynaptic group
        self.n = post_n.shape[0]  # number of neuronsin postsynaptic group
        self.num_synapses = self.m * self.n  # nunmber of synapses to be generated
        self.w_rand_min = w_rand_min  # minimum weight for weight-initialization via uniform distribution
        self.w_rand_max = w_rand_max  # maximum weight for weight-initialization via uniform distribution
        self.initial_w = initial_w  # initial weight value for static weight initialization
        self.enable_wd = enable_wd  # True to enable weight decay

        if weight_multiplier is None:
            self.weight_multiplier = np.ones((self.m, self.n), dtype=np.float)
        else:
            self.weight_multiplier = weight_multiplier
        self.w = self.construct_weights(loaded_weights)  # construct weight matrix

        self.num_histories = int(self.synp.spike_window / self.tki.dt())  # number of discretized time bins that we will keep track of presynaptic spikes in
        self.hist_range = range(self.num_histories)  # just pre-calculating this range for speed
        self.history = np.zeros((self.num_histories, self.m), dtype=np.float)  # A num_histories * num_synapses matrix containing spike counts for each synapse, for each time evaluation step
        self.last_history_update_time = -1.0  # this is the time at which the history array was last updated
        self.p_term = self.precalculate_term()  # precalculate part of the synaptic current calculation
        self.localized_normalization = localized_normalization  # If True, whenever normalize_weights is called, only weights that are enabled as modifiable via the weight multiplier will be normalized

        # NOTE: Experimental, for storing map pointing from neuron index to active indices in weight multipler, used for weight sharing
        self.row_maps = []
        self.share_weights = share_weights  # NOTE: Experimental

    """ Methods that should NOT be overridden -----------------------------------------------------------------------------"""
    def decay_weights(self):
        # NOTE: Experimental
        if self.enable_wd:
            self.w = self.synp.wd * self.w

    def set_row_maps(self, new_row_maps):
        # NOTE: Experimental for weight sharing
        self.row_maps = new_row_maps
    
    def copy_from(self, n1_index, n2_index):
        """
        Copy weights from n1 index into n2
        """
        self.w[self.row_maps[n2_index], n2_index] = self.w[self.row_maps[n1_index], n1_index]
    
    def copy_to_all(self, n_index):
        # NOTE: Experimental for weight sharing
        for i in range(self.post_n.n_num):
            self.copy_from(n_index, i)
        
        self.reset()

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
    
    def set_weights(self, weights):
        """
        Set the weights of the synaptic group with a numpy array
        """
        self.w = self.construct_weights(weights)
    
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
        for k in self.hist_range:
            if any_nonzero(self.history[k]):
                p_term_k = self.p_term[:, k][np.newaxis].transpose()
                i_current = np.dot(self.history[k], self.w*p_term_k*v_term)
                i_syn += i_current
        
        return i_syn
    
    def normalize_weights(self):
        """
        Normalize the weights such that they sum to one
        """
        for i in range(self.w.shape[1]):
            if self.localized_normalization:
                unmasked_elements = np.where(self.weight_multiplier[:,i]==1)
                self.w[:,i][unmasked_elements] = self.w[:,i][unmasked_elements] / self.w[:,i][unmasked_elements].sum()
            else:
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
        if self.synp.lati:  # NOTE: Experimental forced lateral inhibition
            self.post_n.lati(fired_neurons)

    def post_fire_notify(self, fired_neurons):
        """
        Notify this synaptic group of post-synaptic neuron spikes. This is used for both updating the spike
        history and runningonline training
        """
        pass


class PairSTDPSynapticGroup(BaseSynapticGroup):
    def __init__(self, pre_n: NeuralGroup, post_n: NeuralGroup, tki, initial_w: float=None, w_rand_min: float=0.0, w_rand_max: float=1.0, trainable: bool=True, 
                 syn_params: SynapseParams=None, weight_multiplier=None, loaded_weights=None, stdp_params: PairSTDPParams=None, localized_normalization=False, share_weights=False, enable_wd=False):
        super().__init__(pre_n, post_n, tki, initial_w=initial_w, w_rand_min=w_rand_min, w_rand_max=w_rand_max, syn_params=syn_params, weight_multiplier=weight_multiplier, loaded_weights=loaded_weights, trainable=trainable, localized_normalization=localized_normalization, share_weights=share_weights, enable_wd=enable_wd)
        
        # STDP parameters are default if None is given
        self.stdpp = PairSTDPParams() if stdp_params is None else stdp_params

        # pair STDP parameters
        self.a_trace = np.zeros(self.w.shape, dtype=np.float)
        self.b_trace = np.zeros(self.w.shape, dtype=np.float)
    
    def reset(self):
        """
        This is for resting the time-based parameters of the synaptic group.
        """
        self.history.fill(0)
        self.a_trace.fill(0)
        self.b_trace.fill(0)

    def pre_fire_notify(self, pre_count):
        """
        Notify this synaptic group of pre-synaptic neuron spikes. This is used for both updating the spike
        history and running pre-spike online STDP training
        """
        f = np.where(pre_count>0.5)

        # increment presynaptic trace (standard stdp)
        self.a_trace[f,:] += 1.0
        
        self.roll_history_and_assign(pre_count)
        if self.trainable:
            self.pair_stdp(pre_count, 'pre') 

    def post_fire_notify(self, post_count):
        """
        Notify this synaptic group of post-synaptic neuron spikes. This is used for both updating the spike
        history and running post-spike online STDP training
        """
        f = np.where(post_count>0.5)

        # increment postsynaptic trace (standard stdp)
        self.b_trace[:,f] += 1.0

        if self.trainable:
            self.pair_stdp(post_count, 'post')  

    def pair_stdp(self, fire_count, fire_time):
        """
        Runs online standard STDP algorithm with multiplicative weight dependence
        @param fired_neurons: The spike count array
        @param fire_time: 'pre' for presynaptic firing and 'post' for postsynaptic firing
        """
        # calculate change in STDP spike trace parameters using Euler's method
        self.a_trace += -1.0 * self.tki.dt() * self.a_trace / self.stdpp.a_tao
        self.b_trace += -1.0 * self.tki.dt() * self.b_trace / self.stdpp.b_tao

        # only need to run this if there were spikes
        if fire_count.sum() > 0:
            # find the indices where there were spikes
            si = np.where(fire_count > 0)
            
            # calculate new weights and stdp parameters based on firing locations
            if fire_time == 'pre':
                # apply weight change, as a function of the postsynaptic trace
                self.w[si,:] += self.stdpp.ba_scale * self.stdpp.lr * self.w[si,:] * self.b_trace[si,:]
            elif fire_time == 'post':
                # apply weight change, as a function of the presynaptic trace
                self.w[:,si] += self.stdpp.ab_scale * self.stdpp.lr * (1.0 - self.w[:,si]) * self.a_trace[:,si]
                if self.share_weights:
                    c = np.count_nonzero(fire_count == fire_count.max())
                    if c==1:  # if we have a single neuron that fired
                        n_index = np.argmax(fire_count)
                    else:  # if we have multiple neurons that fired the same amount, then choose one randomly
                        n_index = np.random.choice(np.where(fire_count == fire_count.max())[0])
                    # copy weights
                    self.copy_to_all(n_index)
            
            self.w = self.w * self.weight_multiplier


class TripletSTDPSynapticGroup(BaseSynapticGroup):
    def __init__(self, pre_n: NeuralGroup, post_n: NeuralGroup, tki, initial_w: float=None, w_rand_min: float=0.0, w_rand_max: float=1.0, trainable: bool=True, 
                 syn_params: SynapseParams=None, weight_multiplier=None, loaded_weights=None, stdp_params: TripletSTDPParams=None, localized_normalization=False, enable_wd=False):
        super().__init__(pre_n, post_n, tki, initial_w=initial_w, w_rand_min=w_rand_min, w_rand_max=w_rand_max, syn_params=syn_params, weight_multiplier=weight_multiplier, loaded_weights=loaded_weights, trainable=trainable, localized_normalization=localized_normalization, enable_wd=enable_wd)
        
        # STDP parameters are default if None is given
        self.stdpp = TripletSTDPParams() if stdp_params is None else stdp_params 

        # triplet STDP parameters 
        self.stdp_r1 = np.zeros(self.w.shape, dtype=np.float)
        self.stdp_r2 = np.zeros(self.w.shape, dtype=np.float)
        self.stdp_o1 = np.zeros(self.w.shape, dtype=np.float)
        self.stdp_o2 = np.zeros(self.w.shape, dtype=np.float)
    
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
        
        # increment triplet stdp triplet presynaptic trace
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
            
            self.w[si,:] = np.clip(self.w[si,:] - self.stdpp.lr * dw, 0.0, 1.0)
        elif fire_time == 'post':
            dw = self.stdp_r1[:,si] * (self.stdpp.a2_plus + self.stdpp.a3_plus * self.stdp_o2[:,si])
            
            self.w[:,si] = np.clip(self.w[:,si] + self.stdpp.lr * dw, 0.0, 1.0)


class DASTDPSynapticGroup(BaseSynapticGroup):
    def __init__(self, pre_n: NeuralGroup, post_n: NeuralGroup, tki, initial_w: float=None, w_rand_min: float=0.0, w_rand_max: float=1.0, trainable: bool=True, syn_params: SynapseParams=None, weight_multiplier=None, loaded_weights=None, stdp_params: DASTDPParams=None, localized_normalization=False, enable_wd=False):
        super().__init__(pre_n, post_n, tki, initial_w=initial_w, w_rand_min=w_rand_min, w_rand_max=w_rand_max, syn_params=syn_params, weight_multiplier=weight_multiplier, loaded_weights=loaded_weights, trainable=trainable, localized_normalization=localized_normalization, enable_wd=enable_wd)

        # STDP parameters are default if None is given
        self.stdpp = DASTDPParams() if stdp_params is None else stdp_params 

        # eligibility traces for pre-post and post-pre spike pairs
        self.ab_et = np.zeros(self.w.shape, dtype=np.float)
        self.ba_et = np.zeros(self.w.shape, dtype=np.float)

        # spike traces for pre and post spikes
        self.a_trace = np.zeros(self.w.shape, dtype=np.float)
        self.b_trace = np.zeros(self.w.shape, dtype=np.float)
    
    def reset(self):
        """
        This is for resting the time-based parameters of the synaptic group.
        """
        self.history.fill(0)
        self.a_trace.fill(0)
        self.b_trace.fill(0)
        self.ab_et.fill(0)
        self.ba_et.fill(0)

    def pre_fire_notify(self, pre_count):
        """
        Notify this synaptic group of pre-synaptic neuron spikes. This is used for both updating the spike
        history and running pre-spike online STDP training
        """
        f = np.where(pre_count>0.5)

        # increment presynaptic trace (standard stdp)
        self.a_trace[f,:] += 1.0
        
        self.roll_history_and_assign(pre_count)
        
        if self.trainable:
            self.update_eligibility(pre_count, 'pre') 

    def post_fire_notify(self, pre_count):
        """
        Notify this synaptic group of post-synaptic neuron spikes. This is used for both updating the spike
        history and running post-spike online STDP training
        """
        f = np.where(pre_count>0.5)

        # increment postsynaptic trace (standard stdp)
        self.b_trace[:,f] += 1.0

        if self.trainable:
            self.update_eligibility(pre_count, 'post')  

    def update_eligibility(self, fire_count, fire_time):
        """
        Updates the elibiility trace
        @param fire_count: The spike count array
        @param fire_time: 'pre' for presynaptic firing and 'post' for postsynaptic firing
        """
        # calculate change in STDP spike trace parameters using Euler's method
        self.a_trace += -1.0 * self.tki.dt() * self.a_trace / self.stdpp.a_tao
        self.b_trace += -1.0 * self.tki.dt() * self.b_trace / self.stdpp.b_tao

        # calculate change in eligibility traces using Euler's method
        self.ab_et += -1.0 * self.tki.dt() * self.ab_et / self.stdpp.ab_et_tao
        self.ba_et += -1.0 * self.tki.dt() * self.ba_et / self.stdpp.ba_et_tao
        

        # find the indices where there were spikes
        si = np.where(fire_count > 0.5)
        
        # calculate new weights and stdp parameters based on firing locations
        if fire_time == 'pre':
            # increment post-pre eligibility trace proportional to trace left by postsynaptic spikes
            self.ba_et[si,:] += self.b_trace[si,:]
        elif fire_time == 'post':
            # increment pre-post eligibility trace proportional to trace left by presynaptic spikes
            self.ab_et[:,si] += self.a_trace[:,si]

    def apply_dopamine(self, reward, actions=None):
        """
        Apply weight changes based on eligibility traces and reward

        Note: If reward is negative, then STDP will become inversed
        """
        if reward >= 0.0:
            ab_scale = self.stdpp.ab_scale_pos
            ba_scale = self.stdpp.ba_scale_pos
        else:
            ab_scale = self.stdpp.ab_scale_neg
            ba_scale = self.stdpp.ba_scale_neg
        
        # perform pre-post plasticity based on reward
        self.w += self.weight_multiplier * self.ab_et * self.stdpp.lr * ab_scale * np.abs(reward)
        # perform post-pre plasticity based on reward
        self.w += self.weight_multiplier * self.ba_et * self.stdpp.lr * ba_scale * np.abs(reward)

        # make sure that weights stay in bounds
        self.w = np.clip(self.w, 1.0e-5, 1.0)

        # reset eligibility traces
        self.ab_et.fill(0)
        self.ba_et.fill(0)


class InhibitorySynapticGroup(BaseSynapticGroup):
    def __init__(self, pre_n, post_n, tki, initial_w=None, w_rand_min=0.0, w_rand_max=1.0, syn_params=None, weight_multiplier=None, loaded_weights=None, localized_normalization=False, enable_wd=False):
        super().__init__(pre_n, post_n, tki, initial_w=initial_w, w_rand_min=w_rand_min, w_rand_max=w_rand_max, syn_params=syn_params, weight_multiplier=weight_multiplier, loaded_weights=loaded_weights, trainable=False, localized_normalization=localized_normalization, enable_wd=enable_wd)
        # precalculate part of the synaptic current calculation, this is the only change apart from the BaseSynapticGroup initialization, due to having to set the gbar to the inhibitory gbar, regardless of whether or not the NeuralGroup is inhibitory
        self.p_term = self.precalculate_term()

    def calc_isyn(self):
        """
        Calculate the current flowing across this synaptic group, as a function of the spike history.
        The only difference between this one and the BaseSynapticGroup is that in this implementation, the presynaptic reversal potential is ALWAYS the inhibitory reversal potential. This allows us to bypass the need for an extra neuralgroup
        for self-inhibition
        """
        v_m_post = np.zeros((self.m, self.n), dtype=np.float)
        v_rev_pre = np.zeros((self.m, self.n), dtype=np.float)

        v_m_post[:] = self.post_n.v_m
        v_rev_pre.T[:] = self.pre_n.vrev_i
        v_term = v_rev_pre-v_m_post

        i_syn = np.zeros(self.n, dtype=np.float)
        for k in range(self.num_histories):
            hist_k = self.history[k]
            p_term_k = self.p_term[:, k][np.newaxis].transpose()
            i_current = np.dot(hist_k, self.w*p_term_k*v_term)
            i_syn += i_current
        
        return i_syn
    
    def precalculate_term(self):
        delta_t = np.zeros(self.num_histories, dtype=float)
        times = np.arange(1, self.num_histories+1, 1) * self.tki.dt()
        for idx, val in np.ndenumerate(times):
            delta_t[idx] = val

        alpha_time = delta_t / self.synp.tao_syn_i * np.exp(-1.0 * delta_t / self.synp.tao_syn_i)

        gbar_pre = np.zeros(self.m, dtype=np.float).transpose()
        gbar_pre[:] = self.pre_n.params.gbar_i

        return np.outer(gbar_pre.transpose(), alpha_time)

class LatiSynapticGroup:
    """
    Defines a groups of synapses connecting two groups of neurons
    """
    def __init__(self, pre_n: NeuralGroup, post_n: NeuralGroup):
        if pre_n.field_shape != post_n.field_shape:
            raise ValueError("The shapes of the pre and postsynaptic NeuralGroups must be equal but are: %s and %s" % (str(pre_n.field_shape), str(post_n.field_shape)))
        self.trainable = False
        self.pre_n = pre_n  # presynaptic neural group
        self.post_n = post_n  # postsynaptic neural group

        self.m = pre_n.shape[0]  # number of neurons in presynaptic group
        self.n = post_n.shape[0]  # number of neuronsin postsynaptic group

        self.isyn = np.zeros(self.n, dtype=np.float)

    def calc_isyn(self):
        """
        Unused filler
        """
        return self.isyn

    def reset(self):
        """
        Unused filler
        """
        pass

    def pre_fire_notify(self, fired_neurons):
        """
        Perform lateral inhibition
        """
        self.post_n.lati(fired_neurons)

    def post_fire_notify(self, fired_neurons):
        """
        Unused filler
        """
        pass

    def set_weights(self, w):
        """
        Unused filler
        """
        pass
    
    def save_weights(self, p):
        """
        Unused filler
        """
        pass
