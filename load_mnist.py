import pickle
import gzip
import os

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def load():
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, 'data', 'mnist.pkl.gz')
    with gzip.open(filename, 'rb') as ff:
        u = pickle._Unpickler(ff)
        u.encoding = 'latin1'
        train, val, test = u.load()
    return train, val, test

def display(set, i=0):
    image = np.reshape(set[0][i], (28, 28))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    plt.show()

if __name__ == '__main__':
    train, val, test = load()
    display(train)