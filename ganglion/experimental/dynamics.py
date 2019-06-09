import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import LIFNeuralGroup, SensoryNeuralGroup, HomeostaticNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import HomeostaticLIFParams
from units import *
from utils import poisson_train, calculate_phi
import numpy as np
import numpy.random as nprand
import time
import random
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__":
    tki = TimeKeeperIterator(timeunit=0.5*msec)
    duration = 1000 * msec

    p = HomeostaticLIFParams()

    input_layer = SensoryNeuralGroup(np.ones(8, dtype=np.int), "1", tki, n_params)