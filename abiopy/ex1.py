import sys
from timekeeper import TimeKeeperIterator
from NeuralGroup import NeuralGroup
from NeuralNetwork import NeuralNetwork
from Neuron import SpikingNeuron, NeuronParams
from Synapse import Synapse
from units import *
import random
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *
import sys
import numpy as np


if __name__ == "__main__":
    tki = TimeKeeperIterator(timeunit=0.01*msec)
    my_np = NeuronParams()

    g1 = NeuralGroup(0, 10, "input", neuron_params=my_np)
    g2 = NeuralGroup(0, 10, "hidden", neuron_params=my_np)
    g3 = NeuralGroup(0, 1, "output", neuron_params=my_np)
    g1.track_vars(['q_t', 'v_m', 's_t'])
    g2.track_vars(['q_t', 'v_m', 's_t'])
    g3.track_vars(['q_t', 'v_m', 's_t'])

    nn = NeuralNetwork([g1, g2, g3], "my_net")
    nn.fully_connect("input", "hidden")
    nn.fully_connect("hidden", "output")

    duration = 100 * msec
    input_period = 0.1 * msec

    lts = 0
    for step in tki:
        if (step - lts)*tki.dt() >= input_period:
            lts = step
            # add some random inputs to the first neuron group
            g1.dci(np.random.randint(2, size=g1.n_num))
             
        nn.run_order(["input", "hidden", "output"], tki)

        if step >= duration/tki.dt():
            break
    
    n = g1.n[0]
    
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
    # if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
    #     pg.QtGui.QApplication.exec_()
    # f, axarr = plt.subplots(3, sharex=True)
    # f.suptitle('Single Neuron')
    # for n in g2.n:
    #     axarr[0].plot(n.voltage_track)
    #     axarr[1].plot(n.current_track)
    #     axarr[2].plot(n.spike_track)
    # plt.show()