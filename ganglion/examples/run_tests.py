import sys
sys.path.append("../base")

from NeuralGroup import *
from NeuralNetwork import NeuralNetwork
from timekeeper import TimeKeeperIterator
from parameters import *
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

if 6.65152e-07 > tsc > 6.65150e-07:
    test = "PASSED"
else:
    test = "FAILED"
print("Test  ::  %s  ::  Total synaptic current  ::  %g  ::  Expected synaptic current  ::  %s  ::  Execution Time  ::  %g seconds" % (test, np.sum(g3.isyn_track), "6.65151e-07", end_time-start_time))

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

if 4.26568e-05 > tsc > 4.26566e-05:
    test = "PASSED"
else:
    test = "FAILED"
print("Test  ::  %s  ::  Total synaptic current  ::  %g  ::  Expected synaptic current  ::  %s  ::  Execution Time  ::  %g seconds" % (test, np.sum(g3.isyn_track), "4.26567e-05", end_time-start_time))

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

if  8.53136e-05 > tsc >  8.53134e-05:
    test = "PASSED"
else:
    test = "FAILED"
print("Test  ::  %s  ::  Total synaptic current  ::  %g  ::  Expected synaptic current  ::  %s  ::  Execution Time  ::  %g seconds" % (test, np.sum(g3.isyn_track), "8.53135e-05", end_time-start_time))

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

if 6.70269e-08 > tsc > 6.70267e-08:
    test = "PASSED"
else:
    test = "FAILED"
print("Test  ::  %s  ::  Total synaptic current  ::  %g  ::  Expected synaptic current  ::  %s  ::  Execution Time  ::  %g seconds" % (test, np.sum(g3.isyn_track), "6.70268e-08", end_time-start_time))

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

if 6.22392e-08 > tsc > 6.22390e-08:
    test = "PASSED"
else:
    test = "FAILED"
print("Test  ::  %s  ::  Total synaptic current  ::  %g  ::  Expected synaptic current  ::  %s  ::  Execution Time  ::  %g seconds" % (test, np.sum(g3.isyn_track), "6.22391e-08", end_time-start_time))

# -------------------------------------------------------------------------------------------------------

print()
# print("Running Test :: Pair STDP Network...")
# print()
# tki = TimeKeeperIterator(timeunit=0.1*msec)
# duration = 500 * msec

# g1 = SensoryNeuralGroup(1, 4, "input", 1, tki, LIFParams())
# g2 = HSAMLIFNeuralGroup(1, 4, "hidden", 2, tki, HSAMLIFParams())
# g3 = HSAMLIFNeuralGroup(1, 4, "output", 3, tki, HSAMLIFParams())
# g3.tracked_vars = ['i_syn']

# nn = NeuralNetwork([g1, g2, g3], "network", tki)

# nn.fully_connect("input", "hidden", s_type='pair', w_i=0.1)
# nn.fully_connect("hidden", "output", s_type='pair', w_i=0.1)

# start_time = time.time()
# for step in tki:
#     g1.run(np.array([1,0,1,0]).reshape(4, 1))

#     nn.run_order(["input", "hidden", "output"])

#     if step >= duration/tki.dt():
#         break
# end_time = time.time()
# tsc = np.sum(g3.isyn_track)

# if 4.26568e-05 > tsc > 4.26566e-05:
#     test = "PASSED"
# else:
#     test = "FAILED"
# print("Test  ::  %s  ::  Total synaptic current  ::  %g  ::  Expected synaptic current  ::  %s  ::  Execution Time  ::  %g seconds" % (test, np.sum(g3.isyn_track), "4.26567e-05", end_time-start_time))