from utils import load_mnist
from timekeeper import TimeKeeperIterator
from NeuralGroup import NeuralGroup, StructuredNeuralGroup, weight_map_between
from NeuralNetwork import NeuralNetwork
from Neuron import NeuronParams, SpikingNeuron
from units import *
import random
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *


# load MNIST
train_data, train_labels, test_data, test_labels = load_mnist()

params = NeuronParams()

params.membrane_time_constant = 30.0 * msec
params.v_hyperpolarization = -90.0 * mvolt

# set up SNN time tracker
tki = TimeKeeperIterator(timeunit=0.1*msec)
duration = 100000 * msec
input_period = 1.0 * msec

lgn = StructuredNeuralGroup(np.ones((28, 28)), 'lgn', neuron_params=params)
v1_exc = StructuredNeuralGroup(np.ones((5, 5)), 'v1_exc', neuron_params=params)
v1_inh = StructuredNeuralGroup(np.zeros((5, 5)), 'v1_inh', neuron_params=params)

v1_inh.track_vars(['q_t', 'v_m', 's_t'])
v1_exc.track_vars(['q_t', 'v_m', 's_t'])

nn = NeuralNetwork([lgn, v1_exc, v1_inh], "NN")
nn.fully_connect("lgn", "v1_exc")
nn.one_to_one("v1_exc", "v1_inh", w_i=1.1, trainable=False)
nn.fully_connect("v1_inh", "v1_exc", skip_self=True, w_i=1.1, trainable=False)


# clear stdout
os.system('cls' if os.name == 'nt' else 'clear')

lts = 0
lts2 = 0
mnist_counter = 0
for step in tki:
    if (step - lts2)*tki.dt() >= 100*input_period:
        mnist_counter += 1
    if (step - lts)*tki.dt() >= input_period:
        lts = step
        inp = train_data[mnist_counter] * input_period / (17.0*msec)
        if mnist_counter % 2 != 0:
            lgn.dci(inp)
            
    nn.run_order(["lgn", "v1_exc", "v1_inh"], tki, lr_ex=0.01, lr_inh=0.01)

    if step >= duration/tki.dt():
        break
    sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))
print("\n\n")
arr = []
for neuron in v1_exc.n:
    wmap = weight_map_between(lgn, neuron)
    arr.append(wmap)
arr = np.array(arr)
nr = v1_exc.shape[0]
nc = v1_exc.shape[1]
for row in range(nr):
    row_img = np.hstack(arr[row*nc:row*nc +nc, :, :])
    if row == 0:
        img = row_img
    else:
        img = np.vstack((img, row_img))

plt.imshow(img)
plt.show()

n = v1_exc.n[0]

app = QtGui.QApplication(sys.argv)

win = QtGui.QMainWindow()
area = DockArea()
win.setCentralWidget(area)
win.resize(2000,1000)
win.setWindowTitle('Network Activity')

d1 = Dock("D1")
d2 = Dock("D2")
d3 = Dock("D3")
area.addDock(d1, 'bottom')
area.addDock(d2, 'bottom', d1)
area.addDock(d3, 'bottom', d2)
w1 = pg.PlotWidget(title="Voltage Track")
w1.plot(n.voltage_track, pen=pg.mkPen(color=(39, 141, 141), width=3))
w1.plotItem.showGrid(x=True, y=True, alpha=1)
d1.addWidget(w1)

w2 = pg.PlotWidget(title="Charge Track")
w2.plot(n.charge_track, pen=pg.mkPen(color=(48, 196, 121), width=3))
w2.plotItem.showGrid(x=True, y=True, alpha=1)
w2.setXLink(w1)
d2.addWidget(w2)

w3 = pg.PlotWidget(title="Spike Track")
w3.plot(n.spike_track, pen=pg.mkPen(color=(195, 0, 146), width=3))
w3.plotItem.showGrid(x=True, y=True, alpha=1)
w3.setXLink(w2)
d3.addWidget(w3)

win.show()

sys.exit(app.exec_())