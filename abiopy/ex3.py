from timekeeper import TimeKeeperIterator
from NeuralGroup import NeuralGroup, SquareNeuralGroup
from NeuralNetwork import NeuralNetwork
from Neuron import NeuronParams, SpikingNeuron
from units import *
import random
import numpy as np
import sys
import matplotlib.pyplot as plt

tki = TimeKeeperIterator(timeunit=0.01*msec)
duration = 100 * msec
input_period = 0.05 * msec

on_center = np.array([[1, 1, 1],[1, 0, 1],[1, 1, 1]])

rf1 = SquareNeuralGroup(on_center, "rf1")
rf2 = SquareNeuralGroup(on_center, "rf2")
rf3 = SquareNeuralGroup(on_center, "rf3")
rf4 = SquareNeuralGroup(on_center, "rf4")
rf5 = SquareNeuralGroup(on_center, "rf5")
rf6 = SquareNeuralGroup(on_center, "rf6")
rf7 = SquareNeuralGroup(on_center, "rf7")
rf8 = SquareNeuralGroup(on_center, "rf8")
rf9 = SquareNeuralGroup(on_center, "rf9")
r1 = NeuralGroup(0, 1, "r1")
r2 = NeuralGroup(0, 1, "r2")
r3 = NeuralGroup(0, 1, "r3")
r4 = NeuralGroup(0, 1, "r4")
r5 = NeuralGroup(0, 1, "r5")
r6 = NeuralGroup(0, 1, "r6")
r7 = NeuralGroup(0, 1, "r7")
r8 = NeuralGroup(0, 1, "r8")
r9 = NeuralGroup(0, 1, "r9")
v1 = NeuralGroup(0, 3, "V1")

r1.track_vars(['q_t', 'v_m', 's_t'])

nn = NeuralNetwork([rf1, rf2, rf3, rf4, rf5, rf6, rf7, rf8, rf9, r1, r2, r3, r4, r5, r6, r7, r8, r9, v1], "NN")
nn.fully_connect(0, 9, trainable=False, w_i=0.1)
nn.fully_connect(1, 10, trainable=False, w_i=0.1)
nn.fully_connect(2, 11, trainable=False, w_i=0.1)
nn.fully_connect(3, 12, trainable=False, w_i=0.1)
nn.fully_connect(4, 13, trainable=False, w_i=0.1)
nn.fully_connect(5, 14, trainable=False, w_i=0.1)
nn.fully_connect(6, 15, trainable=False, w_i=0.1)
nn.fully_connect(7, 16, trainable=False, w_i=0.1)
nn.fully_connect(8, 17, trainable=False, w_i=0.1)
nn.fully_connect(9, 18, trainable=False, w_i=0.1)
nn.fully_connect(10, 18)
nn.fully_connect(11, 18)
nn.fully_connect(12, 18)
nn.fully_connect(13, 18)
nn.fully_connect(14, 18)
nn.fully_connect(15, 18)
nn.fully_connect(16, 18)
nn.fully_connect(17, 18)

plt.imshow(rf1.weight_map, aspect='auto')
plt.show()
lts = 0
lts2 = 0
img = np.zeros((5, 5), dtype=np.float)
img[:,np.random.randint(1, 4)] = 1.0
for step in tki:
    if (step - lts2)*tki.dt() >= 30*input_period:
        lts2 = step
        # generate an image with a vertical line on it
        img = np.zeros((5, 5), dtype=np.float)
        img[np.random.randint(1, 4),:] = 1.0
    if (step - lts)*tki.dt() >= input_period:
        lts = step
        rf1.dci(img[0:3, 0:3])
        rf2.dci(img[0:3, 1:4])
        rf3.dci(img[0:3, 2:5])
        rf4.dci(img[1:4, 0:3])
        rf5.dci(img[1:4, 1:4])
        rf6.dci(img[1:4, 2:5])
        rf7.dci(img[2:5, 0:3])
        rf8.dci(img[2:5, 1:4])
        rf9.dci(img[2:5, 2:5])
            
    nn.run_order([0, 9, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16, 8, 17, 9, 18, 10, 18, 11, 18, 12, 18, 13, 18, 14, 18, 15, 18, 16, 18, 17, 18], tki)

    if step >= duration/tki.dt():
        break
plt.imshow(rf1.weight_map, aspect='auto')
plt.show()

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *

# n = r1.n[0]

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
