import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data


def load_mnist():
    """
    Load MNIST dataset through tensorflow into numpy arrays. Images are normalized.
    """
    mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=False)

    # train_data = np.vstack([img.reshape(-1,) for img in mnist_data.train.images])
    train_data = np.array([np.reshape(i, (28, 28)) for i in mnist_data.train.images])
    train_labels = mnist_data.train.labels

    # test_data = np.vstack([img.reshape(-1,) for img in mnist_data.test.images])
    test_data =  np.array([np.reshape(i, (28, 28)) for i in mnist_data.test.images])
    test_labels = mnist_data.test.labels

    return train_data, train_labels, test_data, test_labels
