import numpy as np
import matplotlib.pyplot as plt

def genspikes(r=60, dt=1e-3):
    ta = np.arange(0, 1, dt)

    count = 0
    for t in ta:
        if np.random.random() <= r*dt:
            count += 1
    
    return count

counts = []
for _ in range(100):
    counts.append(genspikes())

print(np.mean(counts))
