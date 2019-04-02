from units import *

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
        self.lr = 0.0001
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
