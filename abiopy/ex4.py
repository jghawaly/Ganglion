from timekeeper import TimeKeeperIterator
from NeuralGroup import NeuralGroup, StructuredNeuralGroup, weight_map_between
from NeuralNetwork import NeuralNetwork
from Neuron import NeuronParams, SpikingNeuron
from units import *
import random
import numpy as np
import sys
import matplotlib.pyplot as plt


tki = TimeKeeperIterator(timeunit=0.1*msec)
duration = 100 * msec
input_period = 0.05 * msec

on_center = np.array([[0, 0, 0],[0, 1, 0],[0, 0, 0]])

lgn = StructuredNeuralGroup(np.ones((5, 5)), 'lgn')
v1_exc = StructuredNeuralGroup(np.ones((10, 10)), 'v1_exc')
v1_inh = StructuredNeuralGroup(np.zeros((10, 10)), 'v1_inh')

v1_exc.track_vars(['q_t', 'v_m', 's_t'])

nn = NeuralNetwork([lgn, v1_exc, v1_inh], "NN")
nn.fully_connect("lgn", "v1_exc")
nn.one_to_one("v1_exc", "v1_inh")
nn.fully_connect("v1_inh", "v1_exc", skip_self=True, trainable=False, w_i=1.0)

plt.imshow(weight_map_between(lgn, v1_exc.neuron((0,0))), aspect='auto')
plt.show()
plt.imshow(weight_map_between(lgn, v1_exc.neuron((1,0))), aspect='auto')
plt.show()
lts = 0
lts2 = 0
img = np.zeros((5, 5), dtype=np.float)
img[:, 2] = 1.0
for step in tki:
    if (step - lts2)*tki.dt() >= 30*input_period:
        lts2 = step
        if random.random() >= 0.5:
            # generate an image with a vertical line on it
            img = np.zeros((5, 5), dtype=np.float)
            # img[:, np.random.randint(1, 4)] = 1.0
            img[:, 2] = 1.0
        else:
            # generate an image with a horizontal line on it
            img = np.zeros((5, 5), dtype=np.float)
            img[2,:] = 1.0
    if (step - lts)*tki.dt() >= input_period:
        lts = step
        lgn.dci(img)
            
    nn.run_order(["lgn", "v1_exc", "v1_inh"], tki, lr_ex=0.01, lr_inh=0.01)

    if step >= duration/tki.dt():
        break

plt.imshow(weight_map_between(lgn, v1_exc.neuron((0,0))), aspect='auto')
plt.show()
plt.imshow(weight_map_between(lgn, v1_exc.neuron((1,0))), aspect='auto')
plt.show()

# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtGui
# from pyqtgraph.dockarea import *

# n = v1_exc.n[1]

# app = QtGui.QApplication(sys.argv)

# win = QtGui.QMainWindow()
# area = DockArea()
# win.setCentralWidget(area)
# win.resize(2000,1000)
# win.setWindowTitle('Network Activity')

# d1 = Dock("D1")
# d2 = Dock("D2")
# d3 = Dock("D3")
# area.addDock(d1, 'bottom')
# area.addDock(d2, 'bottom', d1)
# area.addDock(d3, 'bottom', d2)
# w1 = pg.PlotWidget(title="Voltage Track")
# w1.plot(n.voltage_track, pen=pg.mkPen(color=(39, 141, 141), width=3))
# w1.plotItem.showGrid(x=True, y=True, alpha=1)
# d1.addWidget(w1)

# w2 = pg.PlotWidget(title="Charge Track")
# w2.plot(n.charge_track, pen=pg.mkPen(color=(48, 196, 121), width=3))
# w2.plotItem.showGrid(x=True, y=True, alpha=1)
# w2.setXLink(w1)
# d2.addWidget(w2)

# w3 = pg.PlotWidget(title="Spike Track")
# w3.plot(n.spike_track, pen=pg.mkPen(color=(195, 0, 146), width=3))
# w3.plotItem.showGrid(x=True, y=True, alpha=1)
# w3.setXLink(w2)
# d3.addWidget(w3)

# win.show()

# sys.exit(app.exec_())
