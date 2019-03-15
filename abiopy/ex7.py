from utils import load_mnist
from timekeeper import TimeKeeperIterator
from NeuralGroup import NeuralGroup, StructuredNeuralGroup, weight_map_between
from NeuralNetwork import NeuralNetwork
from Neuron import NeuronParams, SpikingNeuron
from Synapse import STDPParams
from units import *
import random
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *


# define data
train_data = np.array([[[1, 0 ,0]], [[0, 1, 0]],[[0, 0, 1]]], dtype=np.float)

params = NeuronParams()
params.max_q = 3.0 * pcoul
l_params = STDPParams()
l_params.lr_plus = 0.1
l_params.lr_minus = 0.1
l_params.window = 20 * msec
l_params.tao_plus = 5 * msec
l_params.tao_minus = 10 * msec

# set up SNN time tracker
tki = TimeKeeperIterator(timeunit=0.01*msec)
duration = 30000 * msec

lgn = StructuredNeuralGroup(np.ones((1, 3)), 'lgn', neuron_params=params)
v1_exc = StructuredNeuralGroup(np.ones((1, 3)), 'v1_exc', neuron_params=params)
v1_inh = StructuredNeuralGroup(np.zeros((1, 3)), 'v1_inh', neuron_params=params)

v1_inh.track_vars(['q_t', 'v_m', 's_t'])
v1_exc.track_vars(['q_t', 'v_m', 's_t'])

nn = NeuralNetwork([lgn, v1_exc, v1_inh], "NN")
nn.fully_connect("lgn", "v1_exc", learning_params=l_params, minw=0.1, maxw=0.75)
nn.one_to_one("v1_exc", "v1_inh", w_i=100.0, trainable=False, minw=0.1, maxw=0.75)
nn.fully_connect("v1_inh", "v1_exc", skip_self=True, w_i=100.0, trainable=False, minw=0.1, maxw=0.75)

def poisson_train(inp: np.ndarray, dt, r):
    # probability of generating a spike at each location
    p = r*dt*inp
    # sample random numbers
    s = np.random.random(p.shape)
    # spike output
    o = np.zeros_like(inp, dtype=np.float)
    # generate spikes
    o[np.where(s <= p)] = 1.0

    return o


# clear stdout
# os.system('cls' if os.name == 'nt' else 'clear')
for neuron in v1_exc.n:
    wmap = weight_map_between(lgn, neuron)
    plt.imshow(wmap)
    plt.show()
# keep track of the last time that the input image was switched
lts = 0
data_index = 0
for step in tki:
    if (step - lts)*tki.dt() >= 30*msec:
        lts = step
        data_index += 1
        if data_index == 3:
            data_index = 0
    
    d = poisson_train(train_data[data_index], tki.dt(), 1000.0)
    lgn.dci(d)
    nn.run_order(["lgn", "v1_exc", "v1_inh"], tki)

    if step >= duration/tki.dt():
        break
    # sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))
print("\n\n")

for neuron in v1_exc.n:
    wmap = weight_map_between(lgn, neuron)
    plt.imshow(wmap)
    plt.show()

if 0:
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