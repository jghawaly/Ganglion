import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy.random as nprand


''' Dataset Stuff '''

def load_mnist(dataset_dir='../datasets/mnist/'):
    """
    Load MNIST dataset through tensorflow into numpy arrays. Images are normalized.
    """
    mnist_data = input_data.read_data_sets(dataset_dir, one_hot=False)

    # train_data = np.vstack([img.reshape(-1,) for img in mnist_data.train.images])
    train_data = np.array([np.reshape(i, (28, 28)) for i in mnist_data.train.images])
    train_labels = mnist_data.train.labels

    # test_data = np.vstack([img.reshape(-1,) for img in mnist_data.test.images])
    test_data =  np.array([np.reshape(i, (28, 28)) for i in mnist_data.test.images])
    test_labels = mnist_data.test.labels

    return train_data, train_labels, test_data, test_labels


''' Neural Network Stuff '''

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


def calculate_phi(f, tki):
    """
    Calculate phi for homeostatic neuron
    @param f: desired spike frequency in Hz
    @param tki: time keeper iterator instace
    """
    return tki.dt() * f


''' Computer Vision Stuff '''

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


def dog_filter(img, gauss1_size=(3,3), gauss2_size=(5,5), gauss_std1=0, gauss_std2=0):
    # perform Gaussian blurring at two different kernel sizes
    gauss1 = cv2.GaussianBlur(img, gauss1_size, gauss_std1)
    gauss2 = cv2.GaussianBlur(img, gauss2_size, gauss_std2)

    # return the difference of the Gaussians
    return gauss1 - gauss2


def add_noise(img, p=0.1):
    img = img.copy()
    m = img.max()
    # generate noise
    for idx, _ in np.ndenumerate(img):
        if nprand.random() <= p:
            img[idx] = nprand.uniform(0.01, m)
        
    return img


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # test the DOG filter on MNIST

    # load the MNIST dataset
    train_data, train_labels, test_data, test_labels = load_mnist()

    for x in range(1000):
        img = add_noise(train_data[x])
        # img = cv2.resize(img, (16,16), interpolation=cv2.INTER_AREA)
        dog = dog_filter(img)

        concat = np.hstack((img, dog))
        plt.imshow(concat, cmap='gray')
        plt.show()
    exit()
