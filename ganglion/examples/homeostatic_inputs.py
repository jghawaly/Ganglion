import sys
sys.path.append("../base")

from utils import *
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    f = log_filter(0.5)
    plt.imshow(f)
    plt.show()

    smnist = StructuredMNIST((0,1,2,3,4,5,6,7,8,9))
    train, label = smnist.all()
    i=0
    for t in train:
        # t = dog_filter(t)
        # t[t<0.9] = 0.0
        t = apply_log_filter(t, f)
        t = unit_normalize(t)
        print(t.sum())
        # t =cv2.resize(t, (16,16), interpolation=cv2.INTER_AREA)
        plt.imshow(t, cmap='gray')
        plt.show()
        i+=1
        if i==20:
            exit()