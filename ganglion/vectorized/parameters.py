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

        # group behavior
        self.force_wta = False


class LIFParams(IFParams):
    def __init__(self):
        super().__init__()
        # Parameters from Brette and Gerstner (2005).
        self.tao_m = 9.37 * msec


class FTLIFParams(LIFParams):
    def __init__(self):
        super().__init__()
        self.ft_add = 3.0 * mvolt
        self.tao_ft = 100.0 * msec


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

        
class PairSTDPParams:
    def __init__(self):
        # a -----> Pre-spike
        # b -----> Post-spike

        # pair STDP settings
        self.a_tao = 5.0 * msec
        self.b_tao = 5.0 * msec
        self.ab_scale = 0.6
        self.ba_scale = -0.3

        # standard settings
        self.lr = 0.05
        self.stdp_window = 20.0 * msec

class TripletSTDPParams:
    visual_cortex = 0
    hippocampus = 1
    def __init__(self, model=1):
        if model == self.__class__.visual_cortex:
            # triplet STDP settings for visual cortex from Pfister and Gerstner (2006)
            self.a2_plus = 5.0e-10
            self.a3_plus = 6.2e-3
            self.a2_minus = 7.0e-3
            self.a3_minus = 2.3e-4
            self.tao_x = 101.0 * msec
            self.tao_y = 125.0 * msec
            self.tao_plus = 16.8 * msec
            self.tao_minus = 33.7 * msec
        elif model == self.__class__.hippocampus:
            # triplet STDP settings for hippocampus from Pfister and Gerstner (2006)
            self.a2_plus = 6.1e-3
            self.a3_plus = 6.7e-3
            self.a2_minus = 1.6e-3
            self.a3_minus = 1.4e-3
            self.tao_x = 946.0 * msec
            self.tao_y = 27.0 * msec
            self.tao_plus = 16.8 * msec
            self.tao_minus = 33.7 * msec
        else:
            raise RuntimeError("An attempt was made to instantiate the TripletSTDPParams class with an invalid value for the model parameter.")

        # standard settings
        self.lr = 0.05
        self.stdp_window = 20.0 * msec

class DASTDPParams():
    def __init__(self):
        # a -----> Pre-spike
        # b -----> Post-spike

        # blah
        self.motor_excitability = False

        # global learning rate
        self.lr = 0.05

        # time constants of spike-trace decay
        self.a_tao = 5.0 * msec
        self.b_tao = 5.0 * msec

        # time constants of eligbility traces
        self.ab_et_tao = 7.21 * sec   
        self.ba_et_tao = 3.61 * sec

        # # scaling factors for positive rewards
        # self.ab_scale_pos = 1.0
        # self.ba_scale_pos = -1.0

        # # scaling factors for negative rewards
        # self.ab_scale_neg = -1.0
        # self.ba_scale_neg = 1.0

        # scaling factors for positive rewards
        self.ab_scale_pos = 1.0
        self.ba_scale_pos = 0.0

        # scaling factors for negative rewards
        self.ab_scale_neg = 0.0
        self.ba_scale_neg = -1.0

        # common settings
        self.stdp_window = 20.0 * msec
