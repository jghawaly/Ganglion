from Neuron import SpikingNeuron, Synapse
from timekeeper import TimeKeeperIterator
from units import *
import numpy as np
import random
from typing import List


class NeuronGroup:
    def __init__(self, inhib_neurons: int, excit_neurons: int, name):
        self.name = name
        self.i_num = inhib_neurons
        self.e_num = excit_neurons
        self.n_num = inhib_neurons + excit_neurons

        self.n: List[SpikingNeuron] = []
        self._construct()
    
    def _construct(self):
        for i in range(self.i_num):
            self.n.append(SpikingNeuron(SpikingNeuron.inhibitory, group_scope=self.name))
        for i in range(self.e_num):
            self.n.append(SpikingNeuron(SpikingNeuron.excitatory, group_scope=self.name))
    
    def track_vars(self, v):
        for n in self.n:
            n.tracked_vars = v


def connect(g1: NeuronGroup, g2: NeuronGroup):
    for n1 in g1.n:
        for n2 in g2.n:
            if random.random() <= 1.0:
                s = Synapse(0, n1, n2, random.random())
                n1.axonal_synapses.append(s)

if __name__ == "__main__":
    g1 = NeuronGroup(0, 10, "input")

    g2 = NeuronGroup(0, 10, "hidden")

    g3 = NeuronGroup(0, 1, "output")

    connect(g1, g2)
    connect(g2, g3)
    
    tki = TimeKeeperIterator(timeunit=0.1*msec)

    g3.track_vars(['q_t', 'v_m', 's_t'])

    duration = 100 * msec
    input_period = 1.0 * msec

    lts = 0
    for step in tki:
        if (step - lts)*tki.dt() >= input_period:
            lts = step
            # add some random inputs to the first neuron group
            for n in g1.n:
                for x in range(1):
                    r = random.random()
                    if r <= 0.1:
                        n.add_spike({'neuron_type': 1, 'weight': 1.0})
        
                        
        active = []
        for n in g1.n:
            o = n.update(tki.dt(), tki.dt())
            if o > 0:
                active.append(1)
            else:
                active.append(0)
        print(active)
        for n in g2.n:
            n.update(tki.dt(), tki.dt())
        
        for n in g3.n:
            n.update(tki.dt(), tki.dt())

        if step == duration/tki.dt():
            break
    
    n = g3.n[0]
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui
    from pyqtgraph.dockarea import *
    import sys
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

