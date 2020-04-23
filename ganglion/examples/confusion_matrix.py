from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import numpy as np 
import argparse 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a confusion matrix for the MNIST network")
    parser.add_argument('--f', type=str, default='', help='beginning string of predictions')
    args = parser.parse_args()

    y_true = np.load(args.f + 'y_true.npy')
    y_pred = np.load(args.f + 'y_pred.npy')

    conmat = confusion_matrix(y_true, y_pred)

    norm_conf = []
    for i in conmat:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    norm_conmat = np.array(norm_conf)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(norm_conmat, cmap='viridis', interpolation='nearest')

    w, h = conmat.shape

    for x in range(w):
        for y in range(h):
            ax.annotate(str(conmat[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

    cb = fig.colorbar(res)
    digits = '0123456789'
    plt.xticks(range(w), digits[:w])
    plt.yticks(range(h), digits[:h])
    plt.show()
