from units import *


class IFParams:
    def __init__(self):
        # self-defined parameters
        self.refractory_period = 2.0 * msec

        # Parameters from Brette and Gerstner (2005).
        self.v_r = -70.6 * mvolt
        self.v_m = -70.6 * mvolt
        self.v_spike = 40.0 * mvolt
        self.v_thr = -50.4 * mvolt
        self.tao_m = 9.37 * msec
        self.c_m = 281.0 * pfarad

        # synaptic current parameters
        self.gbar_e = 100.0 * nsiem
        self.gbar_i = 100.0 * nsiem
        self.vrev_e = 0.0 * mvolt
        self.vrev_i = -75.0 * mvolt


class LIFParams(IFParams):
    def __init__(self):
        super().__init__()
        # Parameters from Brette and Gerstner (2005).
        self.tao_m = 9.37 * msec


class FTLIFParams(LIFParams):
    def __init__(self):
        super().__init__()
        self.ft_add = 3.0 * mvolt
        self.tao_ft = 244.0 * msec


class ExLIFParams(LIFParams):
    def __init__(self):
        super().__init__()
        self.sf = 2.0 * mvolt
        self.v_rheobase = -50.4 * mvolt
        self.v_thr = -50.4 * mvolt


class AdExParams(ExLIFParams):
    def __init__(self):
        super().__init__()
        # Parameters from Brette and Gerstner (2005).
        self.w = 0.0 * namp  
        self.a = 4.0 * nsiem
        self.b = 0.0805 * namp
        self.tao_w = 144.0 * msec


class SynapseParams:
    def __init__(self):
        self.tao_syn = 5.0 * msec  # this probably needs to be 10 msec
        self.spike_window = 20.0 * msec


class STDPParams:
    def __init__(self):
        # triplet STDP settings
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

        # pair STDP settings
        self.stdp_tao_pre = 2.0 * msec
        self.stdp_tao_post = 2.0 * msec
        self.lr_pre = 0.05
        self.lr_post = 0.05
        self.post_multiplier = 0.6
        self.pre_multipler = -0.3

        # common settings
        self.stdp_window = 20.0 * msec
        