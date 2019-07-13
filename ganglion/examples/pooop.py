from sklearn.datasets import load_digits
dataset = load_digits()
labels = dataset.target
images = dataset.images

import matplotlib.pyplot as plt 
for i in range(images.shape[0]):
    print(labels[i])
    plt.gray() 
    plt.matshow(images[i]/images[i].max()) 
    plt.show()