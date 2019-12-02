import sys
sys.path.append("../base")

from NeuralGroup import LIFNeuralGroup, SensoryNeuralGroup
from NeuralNetwork import NeuralNetwork
from timekeeper import TimeKeeperIterator
from parameters import LIFParams
from units import *
import time
import numpy as np

print()
print("Running Test :: Static Network...")
print()
tki = TimeKeeperIterator(timeunit=0.01*msec)
duration = 5 * msec

g1 = SensoryNeuralGroup(1, 4, "input", 1, tki, LIFParams())
g2 = LIFNeuralGroup(1, 4, "hidden", 2, tki, LIFParams())
g3 = LIFNeuralGroup(1, 4, "output", 3, tki, LIFParams())
g3.tracked_vars = ['i_syn']

nn = NeuralNetwork([g1, g2, g3], "network", tki)

nn.fully_connect("input", "hidden", s_type='base', w_i=0.1)
nn.fully_connect("hidden", "output", s_type='base', w_i=0.1)

start_time = time.time()
for step in tki:
    g1.run(np.array([1,1,1,1]).reshape(4, 1))

    nn.run_order(["input", "hidden", "output"])

    if step >= duration/tki.dt():
        break
end_time = time.time()
tsc = np.sum(g3.isyn_track)

if 1.4e-6 > tsc > 1.3e-6:
    test = "PASSED"
else:
    test = "FAILED"
print("Test  ::  %s  ::  Total synaptic current  ::  %g  ::  Expected synaptic current  ::  %s  ::  Execution Time  ::  %g seconds" % (test, np.sum(g3.isyn_track), "1.3632e-06", end_time-start_time))

# -------------------------------------------------------------------------------------------------------

print()
print("Running Test :: Pair STDP Network...")
print()
tki = TimeKeeperIterator(timeunit=0.1*msec)
duration = 500 * msec

g1 = SensoryNeuralGroup(1, 4, "input", 1, tki, LIFParams())
g2 = LIFNeuralGroup(1, 4, "hidden", 2, tki, LIFParams())
g3 = LIFNeuralGroup(1, 4, "output", 3, tki, LIFParams())
g3.tracked_vars = ['i_syn']

nn = NeuralNetwork([g1, g2, g3], "network", tki)

nn.fully_connect("input", "hidden", s_type='pair', w_i=0.1)
nn.fully_connect("hidden", "output", s_type='pair', w_i=0.1)

start_time = time.time()
for step in tki:
    g1.run(np.array([1,0,1,0]).reshape(4, 1))

    nn.run_order(["input", "hidden", "output"])

    if step >= duration/tki.dt():
        break
end_time = time.time()
tsc = np.sum(g3.isyn_track)

if 0.00077 > tsc > 0.00076:
    test = "PASSED"
else:
    test = "FAILED"
print("Test  ::  %s  ::  Total synaptic current  ::  %g  ::  Expected synaptic current  ::  %s  ::  Execution Time  ::  %g seconds" % (test, np.sum(g3.isyn_track), "0.000765634", end_time-start_time))

# -------------------------------------------------------------------------------------------------------

print()
print("Running Test :: Pair STDP Network with Group-Based Inhibition...")
print()
tki = TimeKeeperIterator(timeunit=0.1*msec)
duration = 500 * msec

g1 = SensoryNeuralGroup(1, 4, "input", 1, tki, LIFParams())
g2 = LIFNeuralGroup(1, 4, "hidden", 2, tki, LIFParams())
g2i = LIFNeuralGroup(1, 4, "hidden_i", 2, tki, LIFParams())
g3 = LIFNeuralGroup(1, 4, "output", 3, tki, LIFParams())
g3.tracked_vars = ['i_syn']

nn = NeuralNetwork([g1, g2, g2i, g3], "network", tki)

nn.fully_connect("input", "hidden", s_type='pair', w_i=0.1)
nn.fully_connect("hidden", "output", s_type='pair', w_i=0.1)
nn.fully_connect("hidden", "hidden_i", s_type='pair', w_i=0.1)
nn.fully_connect("hidden_i", "output", s_type='pair', w_i=0.1)

start_time = time.time()
for step in tki:
    g1.run(np.array([1,0,1,0]).reshape(4, 1))

    nn.run_order(["input", "hidden", "hidden_i", "output"])

    if step >= duration/tki.dt():
        break
end_time = time.time()
tsc = np.sum(g3.isyn_track)

if  0.001607 > tsc >  0.0016:
    test = "PASSED"
else:
    test = "FAILED"
print("Test  ::  %s  ::  Total synaptic current  ::  %g  ::  Expected synaptic current  ::  %s  ::  Execution Time  ::  %g seconds" % (test, np.sum(g3.isyn_track), " 0.00160687", end_time-start_time))

# -------------------------------------------------------------------------------------------------------


print()
print("Running Test :: Triplet STDP Network...")
print()
tki = TimeKeeperIterator(timeunit=0.1*msec)
duration = 500 * msec

g1 = SensoryNeuralGroup(1, 4, "input", 1, tki, LIFParams())
g2 = LIFNeuralGroup(1, 4, "hidden", 2, tki, LIFParams())
g3 = LIFNeuralGroup(1, 4, "output", 3, tki, LIFParams())
g3.tracked_vars = ['i_syn']

nn = NeuralNetwork([g1, g2, g3], "network", tki)

nn.fully_connect("input", "hidden", s_type='triplet', w_i=0.1)
nn.fully_connect("hidden", "output", s_type='triplet', w_i=0.1)

start_time = time.time()
for step in tki:
    g1.run(np.array([1,0,1,0]).reshape(4, 1))

    nn.run_order(["input", "hidden", "output"])

    if step >= duration/tki.dt():
        break
end_time = time.time()
tsc = np.sum(g3.isyn_track)

if 8.71e-07 > tsc > 8.7e-07:
    test = "PASSED"
else:
    test = "FAILED"
print("Test  ::  %s  ::  Total synaptic current  ::  %g  ::  Expected synaptic current  ::  %s  ::  Execution Time  ::  %g seconds" % (test, np.sum(g3.isyn_track), "8.70839e-07", end_time-start_time))

# -------------------------------------------------------------------------------------------------------

print()
print("Running Test :: Triplet STDP Network with Group-Based Inhibition...")
print()
tki = TimeKeeperIterator(timeunit=0.1*msec)
duration = 500 * msec

g1 = SensoryNeuralGroup(1, 4, "input", 1, tki, LIFParams())
g2 = LIFNeuralGroup(1, 4, "hidden", 2, tki, LIFParams())
g2i = LIFNeuralGroup(0, 4, "hidden_i", 2, tki, LIFParams())
g3 = LIFNeuralGroup(1, 4, "output", 3, tki, LIFParams())
g3.tracked_vars = ['i_syn']

nn = NeuralNetwork([g1, g2, g2i, g3], "network", tki)

nn.fully_connect("input", "hidden", s_type='triplet', w_i=0.1)
nn.fully_connect("hidden", "output", s_type='triplet', w_i=0.1)
nn.fully_connect("hidden", "hidden_i", s_type='triplet', w_i=0.1)
nn.fully_connect("hidden_i", "output", s_type='triplet', w_i=0.1)

start_time = time.time()
for step in tki:
    g1.run(np.array([1,0,1,0]).reshape(4, 1))

    nn.run_order(["input", "hidden", "hidden_i", "output"])

    if step >= duration/tki.dt():
        break
end_time = time.time()
tsc = np.sum(g3.isyn_track)

if 7.72e-07 > tsc > 7.71e-07:
    test = "PASSED"
else:
    test = "FAILED"
print("Test  ::  %s  ::  Total synaptic current  ::  %g  ::  Expected synaptic current  ::  %s  ::  Execution Time  ::  %g seconds" % (test, np.sum(g3.isyn_track), "7.71805e-07", end_time-start_time))

# -------------------------------------------------------------------------------------------------------