import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy.random as nprand
import os
from scipy.signal import convolve2d as convolve
from numba import jit, prange


''' Dataset Stuff '''

class StructuredMNIST:
    """ NOTE: This was kind of hacked together and may have issues """
    def __init__(self, pattern, digits=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)):
        self.train_data, self.train_labels, _, _ = load_mnist()
        self.pattern = pattern

        self.digit_indices = {}
        self.digit_counter = {}
        self.digit_total = {}
        for digit in digits:
            self.digit_indices[digit] = np.where(self.train_labels==digit)[0]
            self.digit_counter[digit] = 0
            self.digit_total[digit] = self.digit_indices[digit].shape[0]
        self.pattern_counter = 0
    
    def next(self):
        """ Get the next digit. This will keep looping over the dataset without stopping. The caller should
            set the stopping conditions
        """
        # get the next pattern digit
        p = self.pattern[self.pattern_counter]
        # get the current counter for this digit
        c = self.digit_counter[p]
        # get the index of this digit in the train_data
        i = self.digit_indices[p][c]
        # get the digit
        d = self.train_data[i]
        # get the label
        l = self.train_labels[i]
        # iterate the  digit counter
        self.digit_counter[p] = self.digit_counter[p] + 1 if self.digit_counter[p] + 1 < self.digit_total[p] else 0
        # iterate the pattern counter
        self.pattern_counter = self.pattern_counter + 1 if self.pattern_counter + 1 < len(self.pattern) else 0

        return d, l
    
    def all(self):
        """Get complete re-ordered dataset. Warning, this can be memory intensive"""
        train = []
        label = []

        num_mnist = self.train_data.shape[0]
        count = 0
        while count < num_mnist:
            d, l = self.next()
            train.append(d)
            label.append(l)
            count += 1
        
        train = np.array(train, dtype=np.float)
        label = np.array(label, dtype=np.int)

        return train, label


def load_mnist(dataset_dir='../datasets/mnist/'):
    """
    Load MNIST dataset through tensorflow into numpy arrays. Images are normalized.
    """
    if not "train_data.npy" in os.listdir(dataset_dir):
        mnist_data = input_data.read_data_sets(dataset_dir, one_hot=False)

        train_data = np.array([np.reshape(i, (28, 28)) for i in mnist_data.train.images])
        train_labels = mnist_data.train.labels

        test_data =  np.array([np.reshape(i, (28, 28)) for i in mnist_data.test.images])
        test_labels = mnist_data.test.labels

        # save the binary data so that we don't have to keep using tensorflow in the future
        np.save(os.path.join(dataset_dir, 'train_data.npy'), train_data)
        np.save(os.path.join(dataset_dir, 'train_labels.npy'), train_labels)
        np.save(os.path.join(dataset_dir, 'test_data.npy'), test_data)
        np.save(os.path.join(dataset_dir, 'test_labels.npy'), test_labels)

        return train_data, train_labels, test_data, test_labels
    else:
        # just load the data if we already have the binary saved
        train_data = np.load(os.path.join(dataset_dir, 'train_data.npy'))
        train_labels = np.load(os.path.join(dataset_dir, 'train_labels.npy'))
        test_data = np.load(os.path.join(dataset_dir, 'test_data.npy'))
        test_labels = np.load(os.path.join(dataset_dir, 'test_labels.npy'))

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


@jit(nopython=True, parallel=True, nogil=True)
def log(x, y, sigma):
    # calculates the value of a LOG filter at the given indices (x,y) and sigma
    return -1/(np.pi*np.power(sigma, 4.0)) * (1-(x**2+y**2)/(2* sigma**2))*np.exp(-(x**2+y**2)/(2* sigma**2))


@jit(nopython=True, parallel=True, nogil=True)
def unit_normalize(img):
    if img.max() <= 0:
        return img
    return img / img.sum()


@jit(nopython=True, parallel=True, nogil=True)
def max_normalize(img):
    if img.max() <= 0:
        return img
    return img / img.max()


def log_filter(sigma, size=7):
    # generates a LOG filter for the given sigma and kernel size
    if size % 2 == 0:
        raise ValueError("LOG size must be odd, but is %s" % str(size))
    center = size // 2 +1
    indices = np.abs(np.linspace(1,size,size)-center)
    ix, iy = np.meshgrid(indices, indices, indexing='ij')
    k = log(ix, iy, sigma)

    return k

def apply_log_filter(img, f):
    newimg = convolve(img, f, mode='same', boundary='fill')
    return newimg

def add_noise(img, p=0.1):
    # adds Poisson noise to the given image with the given probability
    img = img.copy()
    m = img.max()
    # generate noise
    for idx, _ in np.ndenumerate(img):
        if nprand.random() <= p:
            img[idx] = nprand.uniform(0.01, m)
        
    return img


''' Other Stuff '''

def goodprint(mat):
    for i in range(mat.shape[0]):
        print(np.array2string(mat[i], max_line_width=np.inf))


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
        # t =cv2.resize(t, (16,16), interpolation=cv2.INTER_AREA)
        plt.imshow(t, cmap='gray')
        plt.show()
        i+=1
        if i==20:
            exit()
