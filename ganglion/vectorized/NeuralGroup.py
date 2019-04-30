import numpy as np
from timekeeper import TimeKeeperIterator
from parameters import AdExParams, LIFParams, ExLIFParams, FTLIFParams, IFParams
from units import *
import random


class NeuralGroup:
    """
    This class is a base template containing only items that are common among most neuron models. It 
    should be overriden, and does not run on its own.
    """
    def __init__(self, n_type: np.ndarray, name: str, tki: TimeKeeperIterator, field_shape=None):
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

        # define the virtual shape of the neural group
        if field_shape is None:
            self.field_shape = self.shape
        else:
            self.field_shape = field_shape

    def run(self):
        """
        Update the state of the neurons.

        This method should be overriden for different neuron models
        """
        return None

    def reset(self):
        return None


class SensoryNeuralGroup(NeuralGroup):
    """
    This class defines a group of "neurons" that are not really neurons, they simply send "spikes"
    when the user tells them too
    """
    def __init__(self, n_type: np.ndarray, name: str, tki: TimeKeeperIterator, params: IFParams, field_shape=None):
        super().__init__(n_type, name, tki, field_shape=field_shape)

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
        if spike_count.shape != self.field_shape:
            raise ValueError("Input spike matrix should be the same shape as the neuron's field matrix but are : %s and %s" % (str(spike_count.shape), str(self.field_shape)))
        
        spike_count = np.reshape(spike_count, self.shape)
        
        self.spike_count = spike_count

        output = np.zeros(self.shape, dtype=np.float)

        output[np.where(self.spike_count > 0)] = self.v_spike[np.where(self.spike_count > 0)]  # generate spikes where they are requested

        if "v_m" in self.tracked_vars:
            self.v_m_track.append(output.copy())
        if "spike" in self.tracked_vars:
            self.spike_track.append(spike_count.copy())

        return output


class IFNeuralGroup(NeuralGroup):
    """
    This class defines a group of Integrate and Fire Neurons
    """
    def __init__(self, n_type: np.ndarray, name: str, tki: TimeKeeperIterator, params: IFParams, field_shape=None):
        super().__init__(n_type, name, tki, field_shape=field_shape)
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
        self.vrev_i = np.full(self.shape, params.vrev_i)
        self.vrev_e = np.full(self.shape, params.vrev_e)

        # construct gbar matrix
        self.gbar = np.zeros(n_type.shape, dtype=np.float)  # conductance of any outgoing synapses from these neurons
        self.gbar[np.where(self.n_type==0)] = params.gbar_i 
        self.gbar[np.where(self.n_type==1)] = params.gbar_e

        # Parameters from Brette and Gerstner (2005).
        self.v_r = np.full(self.shape, params.v_r)  # rest potential
        self.v_m = np.full(self.shape, params.v_m)  # membrane potential
        self.v_spike = np.full(self.shape, params.v_spike)  # spike potential
        self.v_thr = np.full(self.shape, params.v_thr)  # spike threshold potential
        self.c_m = np.full(self.shape, params.c_m)  # membrane capacitance

        # parameter tracks
        self.v_m_track = []
        self.isyn_track = []
        self.spike_track = []
    
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
    
    def run(self, i_syn):
        """
        Update the state of the neurons
        """
        # if we are at a new time step since evaluating the neurons, then clear the spike count matrices
        if self.last_spike_count_update != self.tki.tick_time():
            # self.spike_count = np.zeros(self.shape, dtype=np.int)
            self.spike_count.fill(0)
        
        self.spiked = np.zeros(self.shape, dtype=np.int)

        # mask of neurons not in refractory period
        refrac = self.not_in_refractory()

        # calculate change in membrane potential for neurons not in refractory period
        dvm = self.tki.dt() * (i_syn / self.c_m) * refrac
        # print(dvm)
        self.v_m += dvm

        # find indices of neurons that have fired
        self.spiked = np.where(self.v_m >= self.v_thr)

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
        self.v_m[hp] =self.vrev_i[hp]

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
            self.isyn_track.append(i_syn)
        if "spike" in self.tracked_vars:
            self.spike_track.append(self.spike_count.copy())

        return output


class LIFNeuralGroup(IFNeuralGroup):
    """
    This class defines a group of Leaky Integrate and Fire Neurons
    """
    def __init__(self, n_type: np.ndarray, name: str, tki: TimeKeeperIterator, params: LIFParams, field_shape=None, forced_wta=None):
        super().__init__(n_type, name, tki, params=params, field_shape=field_shape)
        # Parameters from Brette and Gerstner (2005).
        self.tao_m = np.full(self.shape, params.tao_m)  # membrane time constant
    
    def run(self, i_syn):
        """
        Update the state of the neurons
        """
        # where_linear = np.where(self.last_spike_time < self.td)
        # where_one = np.where(self.last_spike_time >= self.td)
        # self.s[where_linear] = self.s[where_linear] / self.td
        # self.s[where_one] = 1.0
        # i_syn *= self.s
        # if we are at a new time step since evaluating the neurons, then clear the spike count matrices
        if self.last_spike_count_update != self.tki.tick_time():
            # self.spike_count = np.zeros(self.shape, dtype=np.int)
            self.spike_count.fill(0)
        
        self.spiked = np.zeros(self.shape, dtype=np.int)

        # mask of neurons not in refractory period
        refrac = self.not_in_refractory()

        # calculate change in membrane potential for neurons not in refractory period
        dvm = self.tki.dt() * (-1*(self.v_m - self.v_r) / self.tao_m + i_syn / self.c_m) * refrac
        # print(dvm)
        self.v_m += dvm

        # find indices of neurons that have fired
        self.spiked = np.where(self.v_m >= self.v_thr)

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
        self.v_m[hp] =self.vrev_i[hp]

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
            self.isyn_track.append(i_syn)
        if "spike" in self.tracked_vars:
            self.spike_track.append(self.spike_count.copy())

        return output


class ExLIFNeuralGroup(LIFNeuralGroup):
    """
    This class defines a group of Leaky Integrate and Fire Neurons
    """
    def __init__(self, n_type: np.ndarray, name: str, tki: TimeKeeperIterator, params: ExLIFParams, field_shape=None):
        super().__init__(n_type, name, tki, params=params, field_shape=field_shape)

        # custom parameters
        self.sf = np.full(self.shape, params.sf)  # slope factor for exponential activation nonlinearity term
        self.v_rheobase = np.full(self.shape, params.v_rheobase)  # rheobase potential
    
    def run(self, i_syn):
        """
        Update the state of the neurons
        """
        # if we are at a new time step since evaluating the neurons, then clear the spike count matrices
        if self.last_spike_count_update != self.tki.tick_time():
            # self.spike_count = np.zeros(self.shape, dtype=np.int)
            self.spike_count.fill(0)
        
        self.spiked = np.zeros(self.shape, dtype=np.int)

        # mask of neurons not in refractory period
        refrac = self.not_in_refractory()

        # calculate change in membrane potential for neurons not in refractory period
        dvm = self.tki.dt() * ((-(self.v_m - self.v_r) + self.sf * np.exp((self.v_m - self.v_rheobase) / self.sf)) / self.tao_m + i_syn / self.c_m) * refrac
        self.v_m += dvm

        # find indices of neurons that have fired
        self.spiked = np.where(self.v_m >= self.v_thr)

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
        self.v_m[hp] =self.vrev_i[hp]

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
            self.isyn_track.append(i_syn)
        if "spike" in self.tracked_vars:
            self.spike_track.append(self.spike_count.copy())

        return output


class FTLIFNeuralGroup(LIFNeuralGroup):
    """
    This class defines a group of Leaky Integrate and Fire Neurons
    """
    def __init__(self, n_type: np.ndarray, name: str, tki: TimeKeeperIterator, params: FTLIFParams, field_shape=None, forced_wta=False):
        super().__init__(n_type, name, tki, params=params, field_shape=field_shape)

        # custom parameters
        self.tao_ft = params.tao_ft  # time constant of floating threshold decay
        self.ft_add = params.ft_add  # amount that the floating threshold increases by at each firing
        self.ft = np.zeros(self.shape, dtype=np.float)  # floating threshold
        self.forced_wta = forced_wta  # True to enable forced winner-take-all dynamic

        # custom tracking parameters
        self.ft_track = []
    
    def wta(self, not_spiked):
        self.v_m[not_spiked] = self.v_r[not_spiked]

    def run(self, i_syn):
        """
        Update the state of the neurons
        """
        # if we are at a new time step since evaluating the neurons, then clear the spike count matrices
        if self.last_spike_count_update != self.tki.tick_time():
            # self.spike_count = np.zeros(self.shape, dtype=np.int)
            self.spike_count.fill(0)
        
        self.spiked = np.zeros(self.shape, dtype=np.int)

        # mask of neurons not in refractory period
        refrac = self.not_in_refractory()

        # calculate change in membrane potential for neurons not in refractory period
        self.v_m += self.tki.dt() * (-1*(self.v_m - self.v_r) / self.tao_m + i_syn / self.c_m) * refrac

        # calculate change in floating threshold
        self.ft += -self.tki.dt() * self.ft / self.tao_ft

        # find indices of neurons that have fired
        self.spiked = np.where(self.v_m >= (self.v_thr + self.ft))
        if self.forced_wta:
            if self.spiked[0].shape[0] > 0:
                not_spiked = np.where(self.v_m < (self.v_thr + self.ft))
                self.wta(not_spiked)

        # add a new spike to the spike count for each neuron that fired
        self.spike_count[self.spiked] += 1

        # update the time at which the spike count array was modified
        self.last_spike_count_update = self.tki.tick_time()

        # modify the last-spike-time for each neuron that fired
        self.last_spike_time[self.spiked] = self.tki.tick_time()

        # update floating threshold for each neuron that fired
        self.ft[self.spiked] += self.ft_add

        # If a substantial synaptic inhibitory current is supplied, the timestep used in the Euler's method may not be sufficient to for estimating the amount of
        # synaptic current coming in on the neurons, this can cause the neuron's membrane potential to pull way below the inhibitory reversal potential.
        # To try and "patch" this, we force the neurons that went too low to the reversal potential, which is the theoretical minimum membrane potential that the
        # neuron can have
        hp = np.where(self.v_m < self.vrev_i)
        self.v_m[hp] =self.vrev_i[hp]

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
            self.isyn_track.append(i_syn)
        if "spike" in self.tracked_vars:
            self.spike_track.append(self.spike_count.copy())
        if "ft" in self.tracked_vars:
            self.ft_track.append(self.ft.copy())

        return output
    

class AdExNeuralGroup(ExLIFNeuralGroup):
    """
    This class defines a group of Adaptive Exponential Integrate and Fire Neurons, as described by Brette and Gerstner (2005). 
    NOTE: This model has problems and should not be used for practical networks
    """
    def __init__(self, n_type: np.ndarray, name: str, tki: TimeKeeperIterator, params: AdExParams, field_shape=None):
        super().__init__(n_type, name, tki, params=params, field_shape=field_shape)
        self.a = np.full(self.shape, params.a)  # a parameter for AdEx model
        self.b = np.full(self.shape, params.b)  # b parameter for AdEx model
        self.tao_w = np.full(self.shape, params.tao_w)  # decay time constant for conductance adaptation parameter for AdEx model
        self.w = np.full(self.shape, params.w)  # conductance adaptation parameter for AdEx model

        # parameter tracks
        self.adap_track = []

    def run(self, i_syn):
        """
        Update the state of the neurons via the Adaptive Exponential Integrate and Fire neuron model
        """
        # if we are at a new time step since evaluating the neurons, then clear the spike count matrices
        if self.last_spike_count_update != self.tki.tick_time():
            # self.spike_count = np.zeros(self.shape, dtype=np.int)
            self.spike_count.fill(0)
        
        self.spiked = np.zeros(self.shape, dtype=np.int)

        # mask of neurons not in refractory period
        refrac = self.not_in_refractory()

        # calculate change in membrane potential for neurons not in refractory period
        dvm = self.tki.dt() * ((-(self.v_m - self.v_r) + self.sf * np.exp((self.v_m - self.v_rheobase) / self.sf)) / self.tao_m + i_syn / self.c_m - self.w / self.c_m) * refrac

        # calculate adaptation change
        dw = self.tki.dt() * ((self.a * (self.v_m - self.v_r) - self.w) / self.tao_w) * refrac

        # update variables
        self.v_m += dvm
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

        # increase the w parameter by b for all fired neurons
        self.w[self.spiked] += self.b[self.spiked]

        # if we are tracking any variables, then append them to their respective lists, Note: This can use a lot of memory and cause slowdowns, so only do this when absolutely necessary
        if "v_m" in self.tracked_vars:
            self.v_m_track.append(output.copy())
        if "i_syn" in self.tracked_vars:
            self.isyn_track.append(i_syn)
        if "spike" in self.tracked_vars:
            self.spike_track.append(self.spike_count.copy())
        if "adap" in self.tracked_vars:
            self.adap_track.append(self.w.copy())

        return output
