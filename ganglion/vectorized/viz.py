from NeuralGroup import HSLIFNeuralGroup, NeuralGroup, SensoryNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import HSLIFParams
from timekeeper import TimeKeeperIterator
from utils import poisson_train
from units import *

import pyglet
from pyglet.gl import *
import time
import numpy as np


class NeuralVis:
    def __init__(self, window_width, window_height, nn: NeuralNetwork, grid_padding=50, layer_gap = 1000):
        g = nn.neural_groups[0]
        self.nn = nn
        self.layer_gap = layer_gap

        self.window_width = window_width
        self.window_height = window_height
        self.grid_padding = grid_padding
        self.usable_width = window_width - (g.field_shape[1] + 1) * grid_padding
        self.usable_height = window_height - (g.field_shape[0] + 1) * grid_padding
        self.grid_width = self.usable_width // g.field_shape[1]
        self.grid_height = self.usable_height // g.field_shape[0]
        self.grid_depth = self.grid_width  # NOTE: This could be non-optimal
    
    def cube(self, origin, batch, fired):
        # bottom face vertices
        cb = [origin[0],                   origin[1],                    origin[2],
              origin[0] + self.grid_width, origin[1],                    origin[2],
              origin[0] + self.grid_width, origin[1] + self.grid_height, origin[2],
              origin[0],                   origin[1] + self.grid_height, origin[2]]
        # top face vertices
        ct = [origin[0],                   origin[1],                    origin[2]+self.grid_depth,
              origin[0] + self.grid_width, origin[1],                    origin[2]+self.grid_depth,
              origin[0] + self.grid_width, origin[1] + self.grid_height, origin[2]+self.grid_depth,
              origin[0],                   origin[1] + self.grid_height, origin[2]+self.grid_depth]
        # side face vertices
        s1 = [cb[0], cb[1], cb[2],
              ct[0], ct[1], ct[2],
              ct[9], ct[10], ct[11],
              cb[9], cb[10], cb[11]]
        s2 = [cb[0], cb[1], cb[2],
              ct[0], ct[1], ct[2],
              ct[3], ct[4], ct[5],
              cb[3], cb[4], cb[5]]
        s3 = [cb[3], cb[4], cb[5],
              ct[3], ct[4], ct[5],
              ct[6], ct[7], ct[8],
              cb[6], cb[7], cb[8]]
        s4 = [cb[6], cb[7], cb[8],
              ct[6], ct[7], ct[8],
              ct[9], ct[10], ct[11],
              cb[9], cb[10], cb[11]]
        
        silver = (59, 65, 73)*4
        yellow = (221, 206, 89)*4
        if fired:
            c = yellow
        else:
            c = silver
        # add bottom face
        batch.add(4, pyglet.gl.GL_QUADS, None, ('v3i', cb), ('c3B', c))
        # add top face
        batch.add(4, pyglet.gl.GL_QUADS, None, ('v3i', ct), ('c3B', c))
        batch.add(4, pyglet.gl.GL_QUADS, None, ('v3i', ct), ('c3B', c))
        # add all 4 sides
        # self.batch.add_indexed(4, pyglet.gl.GL_QUADS, None, (0, 4, 7, 3), ('v3i', c))
        batch.add(4, pyglet.gl.GL_QUADS, None, ('v3i', s1), ('c3B', c))
        # self.batch.add_indexed(4, pyglet.gl.GL_QUADS, None, (0, 4, 5, 1), ('v3i', c))
        batch.add(4, pyglet.gl.GL_QUADS, None, ('v3i', s2), ('c3B', c))
        # self.batch.add_indexed(4, pyglet.gl.GL_QUADS, None, (1, 5, 6, 2), ('v3i', c))
        batch.add(4, pyglet.gl.GL_QUADS, None, ('v3i', s3), ('c3B', c))
        # self.batch.add_indexed(4, pyglet.gl.GL_QUADS, None, (2, 6, 7, 3), ('v3i', c))
        batch.add(4, pyglet.gl.GL_QUADS, None, ('v3i', s4), ('c3B', c))
    
    def draw_grid3d(self):
        batch = pyglet.graphics.Batch()
        i = 0
        for g in self.nn.neural_groups:
            for c in range(g.field_shape[1]):
                for r in range(g.field_shape[0]):
                    spikes = np.reshape(g.spike_count.copy(), g.field_shape)
                    # coordinates of bottom left corner of cube (origin)
                    origin = (c*self.grid_width + (c + 1) * self.grid_padding, r*self.grid_height + (r+1) * self.grid_padding, i * self.layer_gap)
                    if spikes[r, c] > 0:
                        self.cube(origin, batch, True)
                    else:
                        self.cube(origin, batch, False)
            i += 1
        
        batch.draw()
                

class NetworkRunHandler:
    def __init__(self, tki: TimeKeeperIterator):
        self.tki = tki
        p = HSLIFParams()
        p.tao_m = 100 * msec
        g1 = SensoryNeuralGroup(1, 16, 'g1', tki, p, field_shape=(4, 4))
        g2 = HSLIFNeuralGroup(1, 25, 'g2', tki, p, field_shape=(5, 5))
        g3 = HSLIFNeuralGroup(1, 9, 'g3', tki, p, field_shape=(3, 3))
        self.g = (g1, g2, g3)
        self.run_order = ("g1", "g2", "g3")
        self.nn = NeuralNetwork(self.g, 'nn1', tki)

        self.nn.fully_connect('g1', 'g2', trainable=True, s_type='triplet')
        self.nn.fully_connect('g2', 'g3', trainable=True, s_type='triplet')
    
    def run(self):
        # inject spikes into sensory layer
        self.g[0].run(poisson_train(np.random.randint(0, 2, size=(4,4)), self.tki.dt(), 600))

        # run all layers
        self.nn.run_order(self.run_order)

        self.tki.__next__()
        self.nn.normalize_weights()


class Window(pyglet.window.Window):
    def __init__(self, nrh: NetworkRunHandler):
        super().__init__(800, 800)
        self.keys = pyglet.window.key.KeyStateHandler()
        self.p = [-400, -400, -1000*len(nrh.nn.neural_groups)]
        self.r = [0, 0, 0]
        self.nrh = nrh
        self.neu_vis = NeuralVis(self.get_size()[0], self.get_size()[1], self.nrh.nn)
        self.key_down = {'left': False, 'right': False, 'A': False, 'D': False, 'W': False, 'S': False}

        pyglet.clock.schedule_interval(self.update, 1/300)
        glEnable (GL_DEPTH_TEST)

    def on_draw(self):
        if self.key_down['left']:
            # rotate camera left
            self.r[1] += 1
        elif self.key_down['right']:
            # rotate camera right
            self.r[1] -= 1
        elif self.key_down['A']:
            # move camera right
            self.p[0] -= 50
        elif self.key_down['D']:
            # move camera left
            self.p[0] += 50
        elif self.key_down['W']:
            # move camera down
            self.p[1] += 50
        elif self.key_down['S']:
            # move camera up
            self.p[1] -= 50

        self.clear()
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, 1, 0.1, 10000)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glTranslatef(*self.p)
        glRotatef(self.r[1], 0, 1, 0)
        # time.sleep(5
        self.neu_vis.draw_grid3d()

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        
        if scroll_y < 0:
            # zoom in
            self.p[2] += 100
        else:
            # zoom out
            self.p[2] -= 100

    def on_key_release(self, symbol, modifiers):
        if symbol == pyglet.window.key.LEFT:
            self.key_down['left'] = False
        elif symbol == pyglet.window.key.RIGHT:
            self.key_down['right'] = False
        elif symbol == pyglet.window.key.A:
            self.key_down['A'] = False
        elif symbol == pyglet.window.key.D:
            self.key_down['D'] = False
        elif symbol == pyglet.window.key.W:
            self.key_down['W'] = False
        elif symbol == pyglet.window.key.S:
            self.key_down['S'] = False

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.LEFT:
            self.key_down['left'] = True
        elif symbol == pyglet.window.key.RIGHT:
            self.key_down['right'] = True
        elif symbol == pyglet.window.key.A:
            self.key_down['A'] = True
        elif symbol == pyglet.window.key.D:
            self.key_down['D'] = True
        elif symbol == pyglet.window.key.W:
            self.key_down['W'] = True
        elif symbol == pyglet.window.key.S:
            self.key_down['S'] = True
    
    def update(self, dt):
        self.nrh.run()
        
 

if __name__ == '__main__':
    tki = TimeKeeperIterator(0.1)
    nrh = NetworkRunHandler(tki)
    window = Window(nrh)
    pyglet.app.run()
