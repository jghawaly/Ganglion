import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
import cv2


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
    """
    Format an array of data as a Poisson spike train.
    @param inp: Numpy array with values between 0 and 1, each representing the intensity of the the input on that neuron
    @param dt: Integration time in Euler's method, should be smaller than 1 / r
    @param r: maximum firing rate of a given neuron, in Hertz
    """
    # probability of generating a spike at each location
    p = r*dt*inp
    # sample random numbers
    s = np.random.random(p.shape)
    # spike output
    o = np.zeros(inp.shape, dtype=np.float)
    # generate spikes
    o[np.where(s <= p)] = 1.0

    return o


def save_img(path, image, normalize=False):
    """
    Save an image. Useful for saving weight maps
    @param path: Path to file, including filename and extension
    @param image: A numpy array 
    @param normalize: True to normalize the data to between 0 and 255
    """
    if normalize:
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    cv2.imwrite(path, image)
