from NeuralGroup import *
from NeuralNetwork import *
from NetworkRunHandler import *
from parameters import *
from timekeeper import TimeKeeperIterator
from utils import poisson_train
from units import *

import pyglet
from pyglet.gl import *
import time
import numpy as np


class Viz:
    def __init__(self, window_width, window_height, nn: NeuralNetwork, grid_padding=50, layer_gap = 1000, group_gap=500, no_show_inhib=False):
        g = nn.neural_groups[0]
        self.nn = nn
        self.layer_gap = layer_gap
        self.group_gap = group_gap

        self.window_width = window_width
        self.window_height = window_height
        self.grid_padding = grid_padding
        # self.usable_width = window_width - (g.field_shape[1] + 1) * grid_padding
        # self.usable_height = window_height - (g.field_shape[0] + 1) * grid_padding
        self.grid_width = 50
        self.grid_height = 50
        self.grid_depth = 50  
        self.no_show_inhib = no_show_inhib

        # colors
        self.silver = (59, 65, 73)*4
        self.yellow = (221, 206, 89)*4
    
    def cube(self, origin, batch, data):
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
        
        if data[0] == 'spike':
            if data[1]:
                c = self.yellow
            else:
                c = self.silver
        elif data[0] == 'v_m':
            if data[2]:
                c =  self.yellow
            else:
                c = (int(255*data[1]), 0, 0)*4
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
    
    def draw_grid3d(self, mode='spike'):
        batch = pyglet.graphics.Batch()
        for g in self.nn.neural_groups:
            if self.no_show_inhib:
                if g.n_type == 1:
                    w = g.field_shape[1]
                    h = g.field_shape[0]
                    for c in range(w):
                        for r in range(h):
                            spikes = np.reshape(g.spike_count.copy(), g.field_shape)

                            # neighbor_cols = self.nn.get_size_of_left_neighbor_group(g.name)
                            # coordinates of bottom left corner of cube (origin)
                            origin = ((g.viz_layer_pos[1] * self.group_gap) + c*self.grid_width + (c + 1) * self.grid_padding, 
                                    r*self.grid_height + (r+1) * self.grid_padding, 
                                    self.layer_gap * g.viz_layer)
                                
                            if mode == 'spike':
                                self.cube(origin, batch, ('spike', True if spikes[r, c] > 0 else False))
                            if mode == 'v_m':
                                if g.__class__ != SensoryNeuralGroup:
                                    # calculate percent of way to firing threshold
                                    vm = g.v_m.reshape(g.field_shape)[r,c]
                                    
                                    ft = g.v_thr.reshape(g.field_shape)[r, c]
                                    rest = g.v_r.reshape(g.field_shape)[r, c]
                                    dec = np.abs(vm / (ft-rest))
                                    
                                    self.cube(origin, batch, ('v_m', dec, True if spikes[r, c] > 0 else False))
                                else:
                                    self.cube(origin, batch, ('spike', True if spikes[r, c] > 0 else False))
        
        batch.draw()
                

class VizWindow(pyglet.window.Window):
    def __init__(self, nth: NetworkRunHandler, no_show_inhib=False):
        super().__init__(800, 800, resizable=True)
        self.keys = pyglet.window.key.KeyStateHandler()
        self.p = [-800, -800, -6000]
        self.r = [0, 0, 0]
        self.nth = nth
        self.neu_vis = Viz(self.get_size()[0], self.get_size()[1], self.nth.nn, no_show_inhib=no_show_inhib)
        self.key_down = {'left': False, 'right': False, 'up': False, 'down': False, 'A': False, 'D': False, 'W': False, 'S': False}

        self.mode = 'spike'

        pyglet.clock.schedule_interval(self.update, 1/10000)
        glEnable(GL_DEPTH_TEST)
    
    def on_resize(self, width, height):
        super().on_resize(width, height)
        # we meed to maintain aspect ratio on window resize
        ratio = 1
        self.set_size(width, width*ratio)

    def on_draw(self):
        if self.key_down['left']:
            # rotate camera left
            self.r[1] += 1
        elif self.key_down['right']:
            # rotate camera right
            self.r[1] -= 1
        elif self.key_down['up']:
            # rotate camera up
            self.r[0] -= 1
        elif self.key_down['down']:
            # rotate camera down
            self.r[0] += 1
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
        glRotatef(self.r[0], 1, 0, 0)
        # time.sleep(5
        self.neu_vis.draw_grid3d(mode=self.mode)

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
        elif symbol == pyglet.window.key.UP:
            self.key_down['up'] = False
        elif symbol == pyglet.window.key.DOWN:
            self.key_down['down'] = False
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
        elif symbol == pyglet.window.key.UP:
            self.key_down['up'] = True
        elif symbol == pyglet.window.key.DOWN:
            self.key_down['down'] = True
        elif symbol == pyglet.window.key.A:
            self.key_down['A'] = True
        elif symbol == pyglet.window.key.D:
            self.key_down['D'] = True
        elif symbol == pyglet.window.key.W:
            self.key_down['W'] = True
        elif symbol == pyglet.window.key.S:
            self.key_down['S'] = True
        elif symbol == pyglet.window.key.V:
            self.mode = 'v_m'
        elif symbol == pyglet.window.key.P:
            self.mode = 'spike'
    
    def update(self, dt):
        """
        Loops over step() until the episode is over. Return True, metrics when it has finished running the episode AND we have not finished ALL episodes. When all
        episodes are over, it returns False, metrics
        """
        running = self.nth.step()
        


if __name__ == '__main__':
    tki = TimeKeeperIterator(timeunit=0.1*msec)
    g1 = SensoryNeuralGroup(1, 4, "input", 1, tki, LIFParams(), viz_layer_pos=(0,0))
    g2 = LIFNeuralGroup(1, 4, "hidden", 2, tki, LIFParams(), viz_layer_pos=(0,0))
    g2i = LIFNeuralGroup(0, 4, 'hidden_i', 2, tki, LIFParams(), viz_layer_pos=(0, 1))
    g3 = LIFNeuralGroup(1, 4, "output", 3, tki, LIFParams(), viz_layer_pos=(0,0))

    nn = NeuralNetwork([g1, g2, g2i, g3], "network", tki)

    nn.fully_connect("input", "hidden", s_type='pair', w_i=0.1)
    nn.one_to_one_connect('hidden', 'hidden_i', trainable=False, w_i=1.0)
    nn.fully_connect('hidden_i', 'hidden', trainable=False, w_i=1.0, skip_one_to_one=True)
    nn.fully_connect("hidden", "output", s_type='pair', w_i=0.1)

    training_data = np.array([np.random.random((4,1))]*50)
    training_labels = np.array([0]*50)
    network_labels = np.array([0,0,0,0])
    run_order = ('input', 'hidden', 'hidden_i', 'output')

    nth = NetworkRunHandler(nn, 
                       training_data,
                       training_labels,
                       network_labels,
                       run_order, episodes=1000)
    window = VizWindow(nth, no_show_inhib=True)
    pyglet.app.run()
