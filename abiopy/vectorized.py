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
        self.tao_syn = 5.0 * msec
        self.spike_window = 20.0 * msec

        
class NeuralGroup:
    """
    This class is a base template containing only items that are common among most neuron models. It 
    should be overriden, and does not run on its own.
    """
    def __init__(self, n_type: np.ndarray, name: str, tki: TimeKeeperIterator):
        self.name = name
        self.tki = tki
        
        try:
            area = n_type.shape[0] * n_type.shape[1]
        except IndexError:
            area = n_type.shape[0]
        self.n_type = n_type
        self.shape = n_type.shape
        self.num_excitatory = np.count_nonzero(n_type)
        self.num_inhibitory = area - np.count_nonzero(n_type)
        self.tracked_vars = []

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

        # holds the NUMBER OF spikes that occured in the last evaluated time window
        self.spike_count = np.zeros(self.shape, dtype=np.int)
        self.v_spike = params.v_spike

        # construct reversal potential matrix
        self.v_rev = np.zeros(n_type.shape, dtype=np.float)
        self.v_rev[np.where(self.n_type==0)] = params.vrev_i
        self.v_rev[np.where(self.n_type==1)] = params.vrev_e

        # construct gbar matrix
        self.gbar = np.zeros(n_type.shape, dtype=np.float)
        self.gbar[np.where(self.n_type==0)] = params.gbar_i
        self.gbar[np.where(self.n_type==1)] = params.gbar_e

        self.v_m_track = []
    
    def run(self, spike_count):
        if spike_count.shape != self.shape:
            raise ValueError("Input spike matrix should be the same shape as the neuron matrix but are : %s and %s" % (str(spike_count.shape), str(self.shape)))
        self.spike_count = spike_count

        output = np.zeros(self.shape, dtype=np.float)

        output[np.where(self.spike_count > 0)] = self.v_spike

        if "v_m" in self.tracked_vars:
            self.v_m_track.append(output)

        return output


class AdExNeuralGroup(NeuralGroup):
    """
    This class defines a group of Adaptive Exponential Integrate and Fire Neurons, as described by Brette and Gerstner (2005)
    """
    def __init__(self, n_type: np.ndarray, name: str, tki: TimeKeeperIterator, params: AdExParams):
        super().__init__(n_type, name, tki)

        # custom parameters
        self.refractory_period = np.full(self.shape, params.refractory_period)
        # holds boolean array of WHETHER OR NOT a spike occured in the last call to run() (Note: NOT THE LAST TIME STEP)
        self.spiked = np.zeros(self.shape, dtype=np.int)
        # holds the NUMBER OF spikes that occured in the last evaluated time window
        self.spike_count = np.zeros(self.shape, dtype=np.int)
        # holds the times of last spike for each neuron
        self.last_spike_time = np.zeros(self.shape, dtype=np.float)
        # holds the time at which the spike count array was last updated
        self.last_spike_count_update = 0.0

        # construct reversal potential matrix
        self.v_rev = np.zeros(n_type.shape, dtype=np.float)
        self.v_rev[np.where(self.n_type==0)] = params.vrev_i
        self.v_rev[np.where(self.n_type==1)] = params.vrev_e

        # construct gbar matrix
        self.gbar = np.zeros(n_type.shape, dtype=np.float)
        self.gbar[np.where(self.n_type==0)] = params.gbar_i
        self.gbar[np.where(self.n_type==1)] = params.gbar_e

        # Parameters from Brette and Gerstner (2005).
        self.v_r = np.full(self.shape, params.v_r)
        self.v_m = np.full(self.shape, params.v_m)
        self.v_spike = np.full(self.shape, params.v_spike)
        self.w = np.full(self.shape, params.w)
        self.v_thr = np.full(self.shape, params.v_thr)
        self.sf = np.full(self.shape, params.sf)
        self.tao_m = np.full(self.shape, params.tao_m)
        self.c_m = np.full(self.shape, params.c_m)
        self.a = np.full(self.shape, params.a)
        self.b = np.full(self.shape, params.b)
        self.tao_w = np.full(self.shape, params.tao_w)

        # parameters tracks
        self.v_m_track = []

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
        Update the state of the neurons.

        This method should be overriden for different neuron models
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

        self.spiked = np.where(self.v_m >= self.v_thr)
        self.spike_count[self.spiked] += 1
        self.last_spike_count_update = self.tki.tick_time()
        self.last_spike_time[self.spiked] = self.tki.tick_time()

        output = self.v_m.copy()

        output[self.spiked] = self.v_spike[self.spiked]
        self.v_m[self.spiked] = self.v_r[self.spiked]

        if "v_m" in self.tracked_vars:
            self.v_m_track.append(output)

        return output

@njit
def fast_row_roll(val, assignment):
    val[1:,:] = val[0:-1,:]
    val[0] = assignment
    return val

class SynapticGroup:
    # NOTE: Add weight chaange tracking by turning self.w into a property call
    def __init__(self, pre_n: AdExNeuralGroup, post_n: AdExNeuralGroup, tki, initial_w: float=None, w_rand_min: float=0.0, w_rand_max: float=1.0, trainable: bool=True, params: SynapseParams=None):
        self.synp = SynapseParams() if params is None else params
        self.tki = tki
        self.pre_n = pre_n
        self.post_n = post_n
        self.trainable = trainable

        # construct the weight matrix
        m = pre_n.shape[0] * pre_n.shape[1]
        n = post_n.shape[0] * post_n.shape[1]
        self.num_synapses = m * n
        if initial_w is None:
            # self.w = np.random.uniform(low=w_rand_min, high=w_rand_max, size=(m, n))
            self.w = np.random.uniform(low=w_rand_min, high=w_rand_max, size=(self.num_synapses))
        else:
            # self.w = np.full((m, n), initial_w)
            self.w = np.full((self.num_synapses), initial_w)
        
        self.pre_spikes = []
        self.post_spikes = []

        # experimental
        self.num_histories = int(self.synp.spike_window / self.tki.dt())
        self.history = np.zeros((self.num_histories, self.num_synapses), dtype=np.float)
        self.delta_t = self.construct_dt_matrix()

    # experimental
    def construct_dt_matrix(self):
        delta_t = np.zeros(self.history.shape, dtype=float)
        times = np.arange(1, self.num_histories+1, 1) * self.tki.dt()
        for idx, val in np.ndenumerate(times):
            delta_t[idx, :] = val
    
        return delta_t
    
    # experimental
    def roll_history_and_assign(self, assignment):
        self.history = fast_row_roll(self.history, assignment)

    def calc_isyn(self):
        # return np.sum(calc_isyn(self.history, self.w, self.post_n.v_m.ravel(), self.pre_n.v_rev.ravel(), self.pre_n.gbar.ravel(), self.delta_t, self.synp.tao_syn))
        return np.sum(self.history * self.w * (self.post_n.v_m.ravel() - self.pre_n.v_rev.ravel()) * self.pre_n.gbar.ravel() * np.exp(-1.0 * self.delta_t / self.synp.tao_syn))


class NeuralNetwork:
    def __init__(self, groups, name, tki):
        self.tki = tki
        self.name = name
        self.neural_groups = groups
        self.synapses = []
    
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
        g1 = self.g(g1_tag)
        g2 = self.g(g2_tag)

        for s in self.synapses:
            if s.pre_n == g1 and s.post_n == g2:
                return s.w 
        
        return None
        
    def run_order(self, group_order, train=True):
        # Evaluating the inputs into each neuron and generating outputs
        # loop over each NeuronGroup
        for o in group_order:
            g = self.g(o)

            # sensory neural groups are treated differently since they do not have any input synapses
            if type(g) == SensoryNeuralGroup:
                # send out spikes to outgoing synapses
                for s2 in self.synapses:
                    if s2.pre_n == g:
                        s2.roll_history_and_assign(g.spike_count) 
            else:
                for s in self.synapses:
                    # if this synapse group is presynaptic, then calculate the current coming across it and run that current through the current neural group
                    if s.post_n == g:
                        i = s.calc_isyn()
                        g.run(i)
                        
                        # send out spikes to outgoing synapses
                        for s2 in self.synapses:
                            if s2.pre_n == g:
                                s2.roll_history_and_assign(g.spike_count)

        
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
        g1 = SensoryNeuralGroup(np.ones((1,3), dtype=np.int), "Luny", tki, AdExParams())
        g2 = AdExNeuralGroup(np.ones((1,1), dtype=np.int), "George", tki, AdExParams())
        g3 = AdExNeuralGroup(np.ones((1,2), dtype=np.int), "Ada", tki, AdExParams())
        g3.tracked_vars = ["v_m"]

        nn = NeuralNetwork([g1, g2, g3], "blah", tki)
        nn.fully_connect("Luny", "George", w_i=1.0)
        nn.fully_connect("George", "Ada", w_i=1.0)

        vms = []

        for step in tki:
            # inject spikes into sensory layer
            g1.run(poisson_train(np.ones(g1.shape, dtype=np.float), tki.dt(), 64))
            # run all layers
            nn.run_order(["Luny", "George", "Ada"])
            
            # sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

            if step >= duration/tki.dt():
                break
        print(time.time()-start)
        times = np.arange(0,len(g3.v_m_track), 1) * tki.dt() / msec
        v = np.ravel(g3.v_m_track)
        
        plt.plot(times, v)
        plt.title("Voltage Track")
        plt.xlabel("Time (msec)")
        plt.ylabel("Membrane Potential (mvolt)")
        plt.show()

        print(nn.get_w_between("Luny", "George"))
