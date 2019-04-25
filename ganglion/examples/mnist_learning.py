import sys
sys.path.append("../vectorized")

from timekeeper import TimeKeeperIterator
from NeuralGroup import SensoryNeuralGroup, IFNeuralGroup, LIFNeuralGroup, ExLIFNeuralGroup, FTLIFNeuralGroup, AdExNeuralGroup
from NeuralNetwork import NeuralNetwork
from parameters import IFParams, LIFParams, ExLIFParams, FTLIFParams, AdExParams, STDPParams
from units import *
from utils import poisson_train, load_mnist, save_img
import numpy as np
import matplotlib.pyplot as plt
import argparse


def form_full_weight_map(group2, group1_label, group2_label, nn):
    arr = []
    for n in range(group2.shape[0]):
        wmap = nn.get_w_between_g_and_n(group1_label, group2_label, n)
        arr.append(wmap)
    arr = np.array(arr)
    nr = group2.field_shape[0]
    nc = group2.field_shape[1]
    for row in range(nr):
        row_img = np.hstack(arr[row*nc:row*nc +nc, :, :])
        if row == 0:
            img = row_img
        else:
            img = np.vstack((img, row_img))

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simulation of a neural network for classification of MNIST digits.')
    parser.add_argument('model', type=str, help='the neuron model to evaluate')
    parser.add_argument('--duration', type=float, default=500000.0, help='duration of simulation (milliseconds)')
    parser.add_argument('--increment', type=float, default=0.2, help='time increment for numerical integration (milliseconds)')
    parser.add_argument('--exposure', type=float, default=200.0, help='the duration of time that the network is exposed to each training example')
    parser.add_argument('--input_rate', type=float, default=64.0, help='maximum firing rate of input sensory neurons (Hz)')
    parser.add_argument('--grid_size', type=int, default=10, help='length and width of square grid that bars are generated on')
    parser.add_argument('--stdp', type=str, default='triplet', help='form of stdp to use, can be pair or triplet')
    parser.add_argument('--save_increment', type=float, default=-1.0, help='time increment to save an image of the weight map during training (millisecnds)')
    parser.add_argument('--save_dir', type=str, default=".", help='path to directory in which to save weight map images')

    args = parser.parse_args()

    if args.model == 'if':
        model = IFNeuralGroup
        params = IFParams
    elif args.model == 'lif':
        model = LIFNeuralGroup
        params = LIFParams
    elif args.model == 'ftlif':
        model = FTLIFNeuralGroup
        params = FTLIFParams
    elif args.model == 'exlif':
        model = ExLIFNeuralGroup
        params = ExLIFParams
    elif args.model == 'adex':
        model = AdExNeuralGroup
        params = AdExParams
    else:
        raise RuntimeError("%s is not a valid neuron model, must be if, lif, ftlif, exlif, or adex.")

    # Load MNIST data
    train_data, train_labels, test_data, test_labels = load_mnist()

    tki = TimeKeeperIterator(timeunit=args.increment*msec)
    duration = args.duration * msec

    inhib_layer_params = params()
    inhib_layer_params.gbar_i = 2000.0 * nsiem
    inhib_layer_params.tao_m = 100 * msec

    exc_layer_params = params()
    exc_layer_params.gbar_e = 100.0 * nsiem
    exc_layer_params.tao_m = 100 * msec
    exc_layer_params.tao_ft = 1000 * msec

    g1 = SensoryNeuralGroup(np.ones(784, dtype=np.int), "inputs", tki, exc_layer_params, field_shape=(28, 28))
    g2 = model(np.ones(args.grid_size * args.grid_size, dtype=np.int), "exc", tki, exc_layer_params, field_shape=(args.grid_size, args.grid_size))
    g3 = model(np.zeros(args.grid_size * args.grid_size, dtype=np.int), "inh_lateral", tki, inhib_layer_params, field_shape=(args.grid_size, args.grid_size))
    # g3.tracked_vars = ["spike"]

    nn = NeuralNetwork([g1, g2, g3], "mnist_learner", tki)
    lp = STDPParams()
    # lp.lr = 0.001
    # lp.a2_minus = 8.0e-3
    # lp.a3_minus = 3e-4
    nn.fully_connect("inputs", "exc", trainable=True, stdp_params=lp, minw=0.05, maxw=0.3, stdp_form=args.stdp)
    nn.one_to_one_connect("exc", "inh_lateral", w_i=1.0, trainable=False)
    nn.fully_connect("inh_lateral", "exc", w_i=1.0, trainable=False, skip_one_to_one=True)

    vms = []

    # -------------------------------------------------------
    img = form_full_weight_map(g2, "inputs", "exc", nn)
    plt.imshow(img)
    plt.title("Pre-training weight matrix")
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()
    # -------------------------------------------------------

    d = train_data[0]
    last_exposure_step = 0
    last_save_step = 0
    rest = False
    mnist_counter = 0
    
    for step in tki:
        # switch inputs and switch rest state if current exposure duration is greater than the requested time
        if (step - last_exposure_step) * tki.dt() >= args.exposure * msec:
            last_exposure_step = step
            rest = not rest
            # we only want to increment training data if we are NOT skipping this exposure
            if not rest:
                d = train_data[mnist_counter]
                mnist_counter += 1
        
        # if save requested
        if args.save_increment > 0.0:
            # save image at given increment
            if (step - last_save_step) * tki.dt() >= args.save_increment * msec:
                last_save_step = step 
                save_img("%s/%s.bmp" % (args.save_dir, str(mnist_counter)), form_full_weight_map(g2, "inputs", "exc", nn), normalize=True)
        
        # if we are in a rest stage, then let the network rest
        if rest:
            g1.run(np.zeros(g1.field_shape, dtype=np.int))
        else:
            # inject spikes into sensory layer
            g1.run(poisson_train(d, tki.dt(), args.input_rate))
        
        # run all layers
        nn.run_order(["inputs", "exc", "inh_lateral"])
        
        sys.stdout.write("Current simulation time: %g milliseconds\r" % (step * tki.dt() / msec))

        if step >= duration/tki.dt():
            break
    
    print("\n\n")
    
    # -------------------------------------------------------
    img = form_full_weight_map(g2, "inputs", "exc", nn)
    plt.imshow(img)
    plt.title("Post-training weight matrix")
    plt.colorbar()
    plt.clim(0, 1)
    plt.show()
    # -------------------------------------------------------
