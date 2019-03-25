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


def poisson_train(inp: np.ndarray, dt, r):
    # probability of generating a spike at each location
    p = r*dt*inp
    # sample random numbers
    s = np.random.random(p.shape)
    # spike output
    o = np.zeros_like(inp, dtype=np.float)
    # generate spikes
    o[np.where(s <= p)] = 1.0

    return o
    