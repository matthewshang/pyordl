import numpy as np
from util import plt

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

def plot(data, labels, net):
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = net.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=10, cmap=plt.cm.Spectral)

    plt.show()

if __name__ == '__main__':
    data, labels = load(100, 2, 3)
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=20, cmap=plt.cm.Spectral)
    plt.show()