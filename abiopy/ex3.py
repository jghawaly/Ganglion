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
duration = 3000 * msec
input_period = 0.1 * msec

on_center = np.array([[0, 0, 0],[0, 1, 0],[0, 0, 0]])

rf1 = StructuredNeuralGroup(on_center, "rf1")
rf2 = StructuredNeuralGroup(on_center, "rf2")
rf3 = StructuredNeuralGroup(on_center, "rf3")
rf4 = StructuredNeuralGroup(on_center, "rf4")
rf5 = StructuredNeuralGroup(on_center, "rf5")
rf6 = StructuredNeuralGroup(on_center, "rf6")
rf7 = StructuredNeuralGroup(on_center, "rf7")
rf8 = StructuredNeuralGroup(on_center, "rf8")
rf9 = StructuredNeuralGroup(on_center, "rf9")
lgn = StructuredNeuralGroup(np.ones(9), 'lgn')
v1_exc = StructuredNeuralGroup(np.ones(2), 'v1_exc')
v1_inh = StructuredNeuralGroup(np.zeros(2), 'v1_inh')

v1_exc.track_vars(['q_t', 'v_m', 's_t'])

nn = NeuralNetwork([rf1, rf2, rf3, rf4, rf5, rf6, rf7, rf8, rf9, lgn, v1_exc, v1_inh], "NN")
rfw = 1.0
nn.all_to_one("rf1", lgn.n[0], trainable=False, w_i=rfw)
nn.all_to_one("rf2", lgn.n[1], trainable=False, w_i=rfw)
nn.all_to_one("rf3", lgn.n[2], trainable=False, w_i=rfw)
nn.all_to_one("rf4", lgn.n[3], trainable=False, w_i=rfw)
nn.all_to_one("rf5", lgn.n[4], trainable=False, w_i=rfw)
nn.all_to_one("rf6", lgn.n[5], trainable=False, w_i=rfw)
nn.all_to_one("rf7", lgn.n[6], trainable=False, w_i=rfw)
nn.all_to_one("rf8", lgn.n[7], trainable=False, w_i=rfw)
nn.all_to_one("rf9", lgn.n[8], trainable=False, w_i=rfw)
nn.fully_connect("lgn", "v1_exc")
nn.one_to_one("v1_exc", "v1_inh", w_i=1.0)
nn.fully_connect("v1_inh", "v1_exc", skip_self=True, trainable=False, w_i=1.0)

plt.imshow(weight_map_between(lgn, v1_exc.neuron((0,)))[np.newaxis], aspect='auto')
plt.show()
plt.imshow(weight_map_between(lgn, v1_exc.neuron((1,)))[np.newaxis], aspect='auto')
plt.show()
lts = 0
lts2 = 0
img = np.random.uniform(0.01, 0.1, (5, 5))
img[:, 2] = 1.0
vert = True
for step in tki:
    if (step - lts2)*tki.dt() >= 1000*input_period:
        lts2 = step
        if vert:
            print("showing vertical")
            # generate an image with a vertical line on it
            img = np.random.uniform(0.01, 0.1, (5, 5))
            # img[:, np.random.randint(1, 4)] = 1.0
            img[:, 2] = 1.0
        else:
            print("showing horizontal")
            # generate an image with a horizontal line on it
            img = np.random.uniform(0.01, 0.1, (5, 5))
            img[2,:] = 1.0
        vert = not vert
    if (step - lts)*tki.dt() >= input_period:
        if random.random() > 0.5:
            rf1.dci(img[0:3, 0:3])
            rf2.dci(img[0:3, 1:4])
            rf3.dci(img[0:3, 2:5])
            rf4.dci(img[1:4, 0:3])
            rf5.dci(img[1:4, 1:4])
            rf6.dci(img[1:4, 2:5])
            rf7.dci(img[2:5, 0:3])
            rf8.dci(img[2:5, 1:4])
            rf9.dci(img[2:5, 2:5])
        lts = step
            
    nn.run_order(["rf1", "rf2", "rf3", "rf4", "rf5", "rf6", "rf7", "rf8", "rf9", "lgn", "v1_exc", "v1_inh"], tki, lr_ex=0.001, lr_inh=0.001)

    if step >= duration/tki.dt():
        break
plt.imshow(weight_map_between(lgn, v1_exc.neuron((0,)))[np.newaxis], aspect='auto')
plt.show()
plt.imshow(weight_map_between(lgn, v1_exc.neuron((1,)))[np.newaxis], aspect='auto')
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
