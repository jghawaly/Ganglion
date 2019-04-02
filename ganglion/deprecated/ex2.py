import sys
from timekeeper import TimeKeeperIterator
from NeuralGroup import StructuredNeuralGroup
from NeuralNetwork import NeuralNetwork
from Neuron import AdExNeuron, AdExParams
from Synapse import Synapse
from units import *
import random
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.dockarea import *
import sys
import numpy as np

# Simple three layer neural network example

if __name__ == "__main__":
    tki = TimeKeeperIterator(timeunit=0.1*msec)
    my_np = AdExParams()

    g1 = StructuredNeuralGroup(np.ones((1, 10)), "input", neuron_params=my_np)
    g2 = StructuredNeuralGroup(np.ones((1, 10)), "hidden", neuron_params=my_np)
    g3 = StructuredNeuralGroup(np.ones((1, 1)), "output", neuron_params=my_np)
    g3.track_vars(['v_m', 's_t', 'wadex'])

    nn = NeuralNetwork([g1, g2, g3], "my_net")
    nn.fully_connect("input", "hidden", minw=0.001, maxw=0.2)
    nn.fully_connect("hidden", "output", minw=0.001, maxw=0.2)

    duration = 100 * msec
    input_period = 17 * msec

    lts = 0
    for step in tki:
        if (step - lts)*tki.dt() >= input_period:
            lts = step
            # add some random inputs to the first neuron group
            g1.direct_injection(0.5*np.random.random(g1.shape), tki.tick_time(), AdExNeuron.desi)
             
        nn.run_order(["input", "hidden", "output"], tki)
        sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))
        if step >= duration/tki.dt():
            break
    print("\n\n")
    n = g3.n[0]
    
    # app = QtGui.QApplication(sys.argv)

    # win = QtGui.QMainWindow()
    # area = DockArea()
    # win.setCentralWidget(area)
    # win.resize(2000,1000)
    # win.setWindowTitle('Network Activity')

    # d1 = Dock("D1")
    # d3 = Dock("D3")
    # area.addDock(d1, 'bottom')
    # area.addDock(d2, 'bottom', d1)
    # area.addDock(d3, 'bottom', d2)
    # w1 = pg.PlotWidget(title="Voltage Track")
    # w1.plot(n.voltage_track, pen=pg.mkPen(color=(39, 141, 141), width=3))
    # w1.plotItem.showGrid(x=True, y=True, alpha=1)
    # d1.addWidget(w1)

    # w3 = pg.PlotWidget(title="Spike Track")
    # w3.plot(n.spike_track, pen=pg.mkPen(color=(195, 0, 146), width=3))
    # w3.plotItem.showGrid(x=True, y=True, alpha=1)
    # w3.setXLink(w2)
    # d3.addWidget(w3)

    # win.show()

    # sys.exit(app.exec_())
    import matplotlib.pyplot as plt
    if sys.flags.interactive != 1 or not hasattr(pg.QtCore, 'PYQT_VERSION'):
        pg.QtGui.QApplication.exec_()
    f, axarr = plt.subplots(3, sharex=True)
    f.suptitle('Single Neuron')
    for n in g3.n:
        axarr[0].plot(n.voltage_track)
        axarr[1].plot(n.spike_track)
        axarr[2].plot(n.wadex_track)
    plt.show()