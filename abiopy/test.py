import pygame
from NeuronGroup import NeuronGroup, connect

from pygame.locals import *

# NN stuff
g1 = NeuronGroup(0, 10, "input")
g2 = NeuronGroup(0, 5, "hidden")
g3 = NeuronGroup(0, 1, "output")
g3.track_vars(['q_t', 'v_m', 's_t'])

connect(g1, g2)
connect(g2, g3)

class GraphicsNeuron(pygame.sprite.Sprite):
    def __init__(self, radius):
        super(GraphicsNeuron, self).__init__()
        self.surf = pygame.Surface((2*radius, 2*radius))
        pygame.draw.circle(self.surf, (0,0 ,255), (radius,radius), radius)
        self.rect = self.surf.get_rect()

pygame.init()

screen = pygame.display.set_mode((1080, 760))

def draw_n_group(g, num_layers, layer):
    v = int(1080/(num_layers+2))
    h_gap = int(760/(g.n_num+2))
    x = 1
    for n in g.n:
        new_n = GraphicsNeuron(10)
        screen.blit(new_n.surf, (v*layer, x*h_gap))
        x+=1


n = GraphicsNeuron(10)


running = True

while running:
    for event in pygame.event.get():
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
        elif event.type == QUIT:
            running = False

    screen.fill((0,0,0))
    draw_n_group(g1, 3, 1)
    draw_n_group(g2, 3, 2)
    draw_n_group(g3, 3, 3)
    pygame.display.flip()
