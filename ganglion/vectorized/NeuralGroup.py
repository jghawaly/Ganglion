import numpy as np
from timekeeper import TimeKeeperIterator
from parameters import AdExParams, LIFParams, ExLIFParams, FTLIFParams, IFParams, HSLIFParams, FTMLIFParams
from units import *
import random
import tensorflow as tf

inhibitory = 0
excitatory = 1

class NeuralGroup:
    """
    This class is a base template containing only items that are common among most neuron models. It 
    should be overriden, and does not run on its own.
    """
    def __init__(self, n_type: int, num: int, name: str, viz_layer: int, tki: TimeKeeperIterator, field_shape=None, viz_layer_pos=(0,0)):
        self.name = name  # the name of this neuron
        self.tki = tki  # the timekeeper isntance that is shared amongst the entire network
        
        self.n_type = n_type  # type of neurons in this group (excitatory or inhibitory)
        self.n_num = num  # number of neurons in this group
        self.shape = (self.n_num,)  # shape/geometry of this neural group
        self.tracked_vars = []  # variables to be tracked throughout the course of evaluation
        self.viz_layer = viz_layer
        self.viz_layer_pos = viz_layer_pos

        # define the virtual shape of the neural group
        if field_shape is None:
            self.field_shape = (self.shape[0],1)  # WARNING: NOT SURE IF THIS WILL CAUSE ISSUES
        else:
            self.field_shape = field_shape

    def pre_update(self, inputs):
        """
        This is called prior to updating the state of the neurons. Inputs would generally be an array of synaptic currents

        This method should be overridden
        """
        return None

    def update(self, inputs):
        """
        Update the state of the neurons. Inputs would generally be an array of synaptic currents. This should perform 
        all necessary operation required to update self.spiked

        This method should be overridden
        """
        return None
    
    def post_update(self):
        """ 
        This is called after updating the state of the neurons

        This method should be overridden
        """
        return None

    def run(self, inputs):
        """
        This executes the chain of pre_update, update, and post_update commands

        This method should be overridden
        """
        self.pre_update(None)
        self.update(None)
        return self.post_update()

    def reset(self):
        return None


class SensoryNeuralGroup(NeuralGroup):
    """
    This class defines a group of "neurons" that are not really neurons, they simply send "spikes"
    when the user tells them too
    """
    def __init__(self, n_type: int, num: int, name: str, viz_layer: int, tki: TimeKeeperIterator, params: IFParams, field_shape=None, viz_layer_pos=(0,0)):
        super().__init__(n_type, num, name, viz_layer, tki, field_shape=field_shape, viz_layer_pos=viz_layer_pos)

        self.spike_count = np.zeros(self.shape, dtype=np.int)  # holds the NUMBER OF spikes that occured in the last evaluated time window
        self.v_spike = np.full(self.shape, params.v_spike)  # spike potential

        # construct reversal potential matrix
        self.v_rev = np.full(self.shape, params.vrev_e if self.n_type == excitatory else params.vrev_i, dtype=np.float)

        # construct gbar matrix
        self.gbar = np.full(self.shape, params.gbar_e if self.n_type == excitatory else params.gbar_i, dtype=np.float)

        # lists that will contain the membrane voltage track
        self.v_m_track = []
        self.spike_track = []
    
    def pre_update(self, spike_count):
        # check to make sure the input has the same dimensions as the group's shape
        if spike_count.shape != self.field_shape:
            raise ValueError("Input spike matrix should be the same shape as the neuron's field matrix but are : %s and %s" % (str(spike_count.shape), str(self.field_shape)))

        return True
    
    def update(self, spike_count):
        """
        Manually "fire" the neurons at the given input
        """
        self.spike_count = np.reshape(spike_count, self.shape)

    def post_update(self, spike_count):
        if "v_m" in self.tracked_vars:
            output = np.zeros(self.shape, dtype=np.float)
            output[np.where(self.spike_count > 0)] = self.v_spike[np.where(self.spike_count > 0)]  # generate spikes where they are requested
            self.v_m_track.append(output)
        if "spike" in self.tracked_vars:
            self.spike_track.append(spike_count.copy())

    def run(self, spike_count):
        self.pre_update(spike_count)
        self.update(spike_count)
        self.post_update(spike_count)
        return self.spike_count


class IFNeuralGroup(NeuralGroup):
    """
    This class defines a group of Integrate and Fire Neurons
    """
    def __init__(self, n_type: int, num: int, name: str, viz_layer: int, tki: TimeKeeperIterator, params: IFParams, field_shape=None, viz_layer_pos=(0,0)):
        super().__init__(n_type, num, name, viz_layer, tki, field_shape=field_shape, viz_layer_pos=viz_layer_pos)
        # custom parameters
        self.params = params
        self.refractory_period = np.full(self.shape, params.refractory_period)  # refractory period for these neurons
        self.spiked = np.zeros(self.shape, dtype=np.int)  # holds boolean array of WHETHER OR NOT a spike occured in the last call to run() (Note: NOT THE LAST TIME STEP)
        self.spike_count = np.zeros(self.shape, dtype=np.int) # holds the NUMBER OF spikes that occured in the last evaluated time window
        self.last_spike_time = np.zeros(self.shape, dtype=np.float)  # holds the TIMES OF last spike for each neuron
        self.last_spike_count_update = 0.0  # holds the time at which the spike count array was last updated

        # construct reversal potential matrix
        self.v_rev = np.full(self.shape, params.vrev_e if self.n_type == excitatory else params.vrev_i, dtype=np.float)
        self.vrev_i = np.full(self.shape, params.vrev_i)
        self.vrev_e = np.full(self.shape, params.vrev_e)

        # construct gbar matrix
        self.gbar = np.full(self.shape, params.gbar_e if self.n_type == excitatory else params.gbar_i, dtype=np.float)

        # Parameters from Brette and Gerstner (2005).
        self.v_r = np.full(self.shape, params.v_r)  # rest potential
        self.v_m = np.full(self.shape, params.v_m)  # membrane potential
        self.v_spike = np.full(self.shape, params.v_spike)  # spike potential
        self.v_thr = np.full(self.shape, params.v_thr)  # spike threshold potential
        self.c_m = np.full(self.shape, params.c_m)  # membrane capacitance

        # parameter tracks
        self.v_m_track = []
        self.v_thr_track = []
        self.isyn_track = []
        self.spike_track = []

        # group behavior
        self.wta = params.force_wta
    
    def reset(self):
        self.v_m = self.v_r.copy()

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
    
    def pre_update(self, i_syn):
        # if we are at a new time step since evaluating the neurons, then clear the spike count matrices
        if self.last_spike_count_update != self.tki.tick_time():
            # self.spike_count = np.zeros(self.shape, dtype=np.int)
            self.spike_count.fill(0)
        
        self.spiked = np.zeros(self.shape, dtype=np.int)
    
    def update(self, i_syn):
        # mask of neurons not in refractory period
        refrac = self.not_in_refractory()

        # calculate change in membrane potential for neurons not in refractory period
        dvm = self.tki.dt() * (i_syn / self.c_m) * refrac
        # print(dvm)
        self.v_m += dvm

        # find indices of neurons that have fired
        self.spiked = np.where(self.v_m >= self.v_thr)

    def track_vars(self, i_syn):
        if "v_m" in self.tracked_vars:
            # this is a copy of the membrane potential matrix
            output = self.v_m.copy()
            # change the output to the spike voltage for each neuron that fired. Note: this does not affect the actual v_m array, just a copy of it, because
            # in this neuron model, the voltage of a spike does not really have any specific meaning, rather, it is the time of the spikes that matter
            output[self.spiked] = self.v_spike[self.spiked]
            self.v_m_track.append(output.copy())
        if "i_syn" in self.tracked_vars:
            self.isyn_track.append(i_syn)
        if "spike" in self.tracked_vars:
            self.spike_track.append(self.spike_count.copy())
        if "v_thr" in self.tracked_vars:
            self.v_thr_track.append(self.v_thr.copy())

    def post_update(self, i_syn):
        # add a new spike to the spike count for each neuron that fired
        self.spike_count[self.spiked] += 1
        # update the time at which the spike count array was modified
        self.last_spike_count_update = self.tki.tick_time()
        # modify the last-spike-time for each neuron that fired
        self.last_spike_time[self.spiked] = self.tki.tick_time()

        # If a substantial synaptic inhibitory current is supplied, the timestep used in the Euler's method may not be sufficient to for estimating the amount of
        # synaptic current coming in on the neurons, this can cause the neuron's membrane potential to pull way below the inhibitory reversal potential.
        # To try and "patch" this, we force the neurons that went too low to the reversal potential, which is the theoretical minimum membrane potential that the
        # neuron can have
        hp = np.where(self.v_m < self.vrev_i)
        self.v_m[hp] = self.vrev_i[hp]

        # change the actual membrane voltage to the resting potential for each neuron that fired
        self.v_m[self.spiked] = self.v_r[self.spiked]
    
    def run(self, i_syn):
        """
        Update the state of the neurons
        """
        self.pre_update(i_syn)
        self.update(i_syn)
        self.post_update(i_syn)
        self.track_vars(i_syn)

        return self.spike_count


class LIFNeuralGroup(IFNeuralGroup):
    """
    This class defines a group of Leaky Integrate and Fire Neurons
    """
    def __init__(self, n_type: int, num: int, name: str, viz_layer: int, tki: TimeKeeperIterator, params: LIFParams, field_shape=None, forced_wta=None, viz_layer_pos=(0,0)):
        super().__init__(n_type, num, name, viz_layer, tki, params=params, field_shape=field_shape, viz_layer_pos=viz_layer_pos)
        # Parameters from Brette and Gerstner (2005).
        self.tao_m = np.full(self.shape, params.tao_m)  # membrane time constant
    
    def update(self, i_syn):
        # mask of neurons not in refractory period
        refrac = self.not_in_refractory()

        # calculate change in membrane potential for neurons not in refractory period
        dvm = self.tki.dt() * (-1*(self.v_m - self.v_r) / self.tao_m + i_syn / self.c_m) * refrac
        # print(dvm)
        self.v_m += dvm

        # find indices of neurons that have fired
        self.spiked = np.where(self.v_m >= self.v_thr)


class ExLIFNeuralGroup(LIFNeuralGroup):
    """
    This class defines a group of Leaky Integrate and Fire Neurons
    """
    def __init__(self, n_type: int, num: int, name: str, viz_layer: int, tki: TimeKeeperIterator, params: ExLIFParams, field_shape=None, viz_layer_pos=(0,0)):
        super().__init__(n_type, num, name, viz_layer, tki, params=params, field_shape=field_shape, viz_layer_pos=viz_layer_pos)

        # custom parameters
        self.sf = np.full(self.shape, params.sf)  # slope factor for exponential activation nonlinearity term
        self.v_rheobase = np.full(self.shape, params.v_rheobase)  # rheobase potential
    
    def update(self, i_syn):
        # mask of neurons not in refractory period
        refrac = self.not_in_refractory()

        # calculate change in membrane potential for neurons not in refractory period
        dvm = self.tki.dt() * ((-(self.v_m - self.v_r) + self.sf * np.exp((self.v_m - self.v_rheobase) / self.sf)) / self.tao_m + i_syn / self.c_m) * refrac
        self.v_m += dvm

        # find indices of neurons that have fired
        self.spiked = np.where(self.v_m >= self.v_thr)


class FTMLIFNeuralGroup(LIFNeuralGroup):
    def __init__(self, n_type: int, num: int, name: str, viz_layer: int, tki: TimeKeeperIterator, params: FTMLIFParams, field_shape=None, viz_layer_pos=(0,0)):
        super().__init__(n_type, num, name, viz_layer, tki, params=params, field_shape=field_shape, viz_layer_pos=viz_layer_pos)

        # custom parameters
        self.dftm = params.dftm           # percent by which firing threshold changes
        self.tao_ftm = params.tao_ftm     # decay constant
        self.mar = params.min_above_rest  # lowest percent above rest to allow neurons to get to
        self.ftm = np.zeros(self.shape, dtype=np.float)  # floating threshold
    
    def update(self, i_syn):
        # mask of neurons not in refractory period
        refrac = self.not_in_refractory()

        # calculate change in membrane potential for neurons not in refractory period
        self.v_m += self.tki.dt() * (-1*(self.v_m - self.v_r) / self.tao_m + i_syn / self.c_m) * refrac

        # calculate change in floating threshold
        self.ftm += -self.tki.dt() * self.ftm / self.tao_ftm

        # find indices of neurons that have fired
        self.spiked = np.where(self.v_m >= (self.v_thr + self.ftm))
    
    def ftm_mod(self, label, decision, supervised=False):
        # print(label)
        # print(decision)
        # print(self.ftm)
        correct = label==decision
        if not supervised:
            sign = np.full(self.shape, 1.0 if correct else -1.0)
            sign[label] = -1.0 if correct else 1.0
        else:
            sign = np.full(self.shape, 1.0)
            sign[label] = -1.0
        
        # print(sign)
        self.ftm -= self.dftm * sign * self.v_thr
        self.ftm[self.ftm > (self.v_thr - self.v_r)] = (1.0-self.mar) * (self.params.v_thr-self.params.v_r)
        # print(self.ftm)
        # print(self.v_thr + self.ftm)
        
        # l = np.where(self.v_thr - self.v_r >= 0)
        # self.v_thr[l] = self.v_r[l] + self.mar * self.v_r[l]
    
    def reset_ftm(self):
        self.ftm.fill(0.0)


class FTLIFNeuralGroup(LIFNeuralGroup):
    """
    This class defines a group of Leaky Integrate and Fire Neurons with a floating threshold that acts as a firing-rate adaptation parameter
    """
    def __init__(self, n_type: int, num: int, name: str, viz_layer: int, tki: TimeKeeperIterator, params: FTLIFParams, field_shape=None, forced_wta=False, viz_layer_pos=(0,0)):
        super().__init__(n_type, num, name, viz_layer, tki, params=params, field_shape=field_shape, viz_layer_pos=viz_layer_pos)

        # custom parameters
        self.tao_ft = params.tao_ft  # time constant of floating threshold decay
        self.ft_add = params.ft_add  # amount that the floating threshold increases by at each firing
        self.ft = np.zeros(self.shape, dtype=np.float)  # floating threshold
        self.forced_wta = forced_wta  # True to enable forced winner-take-all dynamic

        # custom tracking parameters
        self.ft_track = []

    
    def track_vars(self, i_syn):
        # if we are tracking any variables, then append them to their respective lists, Note: This can use a lot of memory and cause slowdowns, so only do this when absolutely necessary
        if "v_m" in self.tracked_vars:
            # this is a copy of the membrane potential matrix
            output = self.v_m.copy()
            # change the output to the spike voltage for each neuron that fired. Note: this does not affect the actual v_m array, just a copy of it, because
            # in this neuron model, the voltage of a spike does not really have any specific meaning, rather, it is the time of the spikes that matter
            output[self.spiked] = self.v_spike[self.spiked]
            self.v_m_track.append(output.copy())
        if "i_syn" in self.tracked_vars:
            self.isyn_track.append(i_syn)
        if "spike" in self.tracked_vars:
            self.spike_track.append(self.spike_count.copy())
        if "ft" in self.tracked_vars:
            self.ft_track.append(self.ft.copy())
        if "v_thr" in self.tracked_vars:
            self.v_thr_track.append(self.v_thr.copy())

    def update(self, i_syn):
        # mask of neurons not in refractory period
        refrac = self.not_in_refractory()

        # calculate change in membrane potential for neurons not in refractory period
        self.v_m += self.tki.dt() * (-1*(self.v_m - self.v_r) / self.tao_m + i_syn / self.c_m) * refrac

        # calculate change in floating threshold
        self.ft += -self.tki.dt() * self.ft / self.tao_ft

        # find indices of neurons that have fired
        self.spiked = np.where(self.v_m >= (self.v_thr + self.ft))

        # update floating threshold for each neuron that fired
        self.ft[self.spiked] += self.ft_add


class HSLIFNeuralGroup(LIFNeuralGroup):
    """
    This class defines a group of Leaky Integrate and Fire Neurons with Homeostasis that acts to regulate the firing threshold of the neuron in order to maintain 
    the average firing rate within a user-defined window.
    """
    def __init__(self, n_type: int, num: int, name: str, viz_layer: int, tki: TimeKeeperIterator, params: HSLIFParams, field_shape=None, viz_layer_pos=(0,0)):
        super().__init__(n_type, num, name, viz_layer, tki, params=params, field_shape=field_shape, viz_layer_pos=viz_layer_pos)

        # custom parameters
        self.nip = params.nip
        self.phi = np.full(self.shape, params.phi)
    
    def update(self, i_syn):
        # mask of neurons not in refractory period
        refrac = self.not_in_refractory()

        # calculate change in membrane potential for neurons not in refractory period
        dvm = self.tki.dt() * (-1*(self.v_m - self.v_r) / self.tao_m + i_syn / self.c_m) * refrac
        # print(dvm)
        self.v_m += dvm

        # find indices of neurons that have fired
        self.spiked = np.where(self.v_m >= self.v_thr)

        # generate spike mask: Non-standard LIF code starts here -------------------------------------
        s = np.zeros_like(self.spike_count)
        s[self.spiked] = 1.0

        # adjust threshold voltage
        self.v_thr = self.v_thr + self.nip * (s - self.phi)

        # make sure threshold does not drop below resting potential
        where_low = np.where(self.v_thr <= self.v_r)
        self.v_thr[where_low] = 0.99 * self.v_r[where_low]


class AdExNeuralGroup(ExLIFNeuralGroup):
    """
    This class defines a group of Adaptive Exponential Integrate and Fire Neurons, as described by Brette and Gerstner (2005). 
    NOTE: This model has problems and should not be used for practical networks
    """
    def __init__(self, n_type: int, num: int, name: str, viz_layer: int, tki: TimeKeeperIterator, params: AdExParams, field_shape=None, viz_layer_pos=(0,0)):
        super().__init__(n_type, num, name, viz_layer, tki, params=params, field_shape=field_shape, viz_layer_pos=viz_layer_pos)
        self.a = np.full(self.shape, params.a)  # a parameter for AdEx model
        self.b = np.full(self.shape, params.b)  # b parameter for AdEx model
        self.tao_w = np.full(self.shape, params.tao_w)  # decay time constant for conductance adaptation parameter for AdEx model
        self.w = np.full(self.shape, params.w)  # conductance adaptation parameter for AdEx model

        # parameter tracks
        self.adap_track = []
    
    def reset(self):
        self.v_m = self.v_r.copy()
        self.w = np.full(self.shape, self.params.w)

    def update(self, i_syn):
        # mask of neurons not in refractory period
        refrac = self.not_in_refractory()

        # calculate change in membrane potential for neurons not in refractory period
        self.v_m += self.tki.dt() * ((-(self.v_m - self.v_r) + self.sf * np.exp((self.v_m - self.v_rheobase) / self.sf)) / self.tao_m + i_syn / self.c_m - self.w / self.c_m) * refrac

        # calculate adaptation change
        self.w += self.tki.dt() * ((self.a * (self.v_m - self.v_r) - self.w) / self.tao_w) * refrac

        # find indices of neurons that have fired
        self.spiked = np.where(self.v_m >= self.v_thr)

        # increase the w parameter by b for all fired neurons
        self.w[self.spiked] += self.b[self.spiked]

    def track_vars(self, i_syn):
        # if we are tracking any variables, then append them to their respective lists, Note: This can use a lot of memory and cause slowdowns, so only do this when absolutely necessary
        if "v_m" in self.tracked_vars:
            # this is a copy of the membrane potential matrix
            output = self.v_m.copy()
            # change the output to the spike voltage for each neuron that fired. Note: this does not affect the actual v_m array, just a copy of it, because
            # in this neuron model, the voltage of a spike does not really have any specific meaning, rather, it is the time of the spikes that matter
            output[self.spiked] = self.v_spike[self.spiked]
            self.v_m_track.append(output.copy())
        if "i_syn" in self.tracked_vars:
            self.isyn_track.append(i_syn)
        if "spike" in self.tracked_vars:
            self.spike_track.append(self.spike_count.copy())
        if "adap" in self.tracked_vars:
            self.adap_track.append(self.w.copy())
        if "v_thr" in self.tracked_vars:
            self.v_thr_track.append(self.v_thr.copy())
