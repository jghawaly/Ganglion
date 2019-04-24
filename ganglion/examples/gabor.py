import sys
sys.path.append("../vectorized")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_mnist


if __name__ == "__main__":
    # Load MNIST data
    train_data, train_labels, test_data, test_labels = load_mnist()
    
    img = train_data[0]
    num_orientations = 9
    angles = np.linspace(0, np.pi, num_orientations)
    filter_bank = []
    x=0
    for theta in angles:
        filter_bank.append(cv2.getGaborKernel((10,10), 1.0, theta, 1, 0.25, 0))
        if x == 0:
            total_filter = filter_bank[-1]
        else:
            total_filter = np.hstack((total_filter, filter_bank[-1]))
        x += 1
    
    x=0
    for f in filter_bank:
        if x == 0:
            desc = cv2.filter2D(img, -1, f)
        else:
            desc = np.dstack((desc, cv2.filter2D(img, -1, f)))
        x += 1
    
    f = cv2.filter2D(img, -1, filter_bank[0])
    f[np.where(f<10)] = 0
    plt.imshow(total_filter, cmap='gray')
    plt.show()
    plt.imshow(img, cmap='gray')
    plt.show()
    plt.imshow(f, cmap='gray')
    plt.show()
