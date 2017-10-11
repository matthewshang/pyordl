import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def load(num, dims, classes):
    data = np.zeros((num * classes, dims))
    labels = np.zeros(num * classes, dtype='uint8')

    for i in range(classes):
        ix = range(num * i, num * (i+1))
        r = np.linspace(0.0, 1.0, num)
        t = np.linspace(i * 4, (i+1) * 4, num)
        t += np.random.randn(num) * 0.2
        data[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        labels[ix] = i

    return data, labels

if __name__ == '__main__':
    data, labels = load(100, 2, 3)
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=20, cmap=plt.cm.Spectral)
    plt.show()