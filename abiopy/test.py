import numpy as np
import matplotlib.pyplot as plt


vr = -50
v1 = 40
tm = 10

def dv1(volts, dt):
    return volts - (volts-vr)*(1.0 - np.exp(-dt/tm))


def dv2(volts, dt):
    return volts * np.exp(-dt/tm)

one = []
two = []

for x in range(20):
    v1 = dv1(v1, 1)
    one.append(v1)

vr = -50
v1 = 40
tm = 10

for x in range(20):
    v1 = dv2(v1, 1)
    two.append(v1)

plt.plot(one, "r")
plt.plot(two, "b")
plt.legend(("one", "two"))
plt.show()
