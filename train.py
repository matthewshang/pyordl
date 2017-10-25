import numpy as np
from timeit import default_timer as timer
from net import Net
from layers import Linear, Relu, Softmax
import util
from util import mpl, plt
import load_mnist as mnist

def minibatch(data, labels, batch_size=32):
    num_batches = data.shape[0] // batch_size + 1
    for i in range(num_batches):
        idx = i * batch_size
        chunk = (data[idx:(idx + batch_size)], labels[idx:(idx + batch_size)])
        yield chunk

def main():
    train, val, test = mnist.load()
    train_data = train[0]
    train_labels = train[1]
    one_hot = util.to_categorical(train_labels)
    
    reg = 1e-3
    step_size = 1e-1
    batch_size = 64
    epochs = 50
    n = Net()
    n.add(Linear(784, 50))
    n.add(Relu())
    n.add(Linear(50, 10))
    n.add(Softmax())

    # batch_costs = []
    # train_costs = []
    train_accuracies = []
    val_accuracies = []
    
    start = timer()
    for i in range(epochs):
        for data, labels in minibatch(train_data, one_hot, 128):
            out = n.forward(data)
            loss = n.backprop(labels)     
            n.update(step_size)

        val_accuracy = np.mean(n.predict(val[0]) == val[1])
        val_accuracies.append(val_accuracy)

        train_accuracy = np.mean(n.predict(train[0]) == train[1])
        train_accuracies.append(train_accuracy)

        if i % 10 == 0:
            print("iter {} cost: {}".format(i, loss))
    
    total_time = timer() - start
    print('total time: {}s, time per epoch: {}s'.format(total_time, total_time / epochs))
    accuracy = np.mean(n.predict(test[0]) == test[1])
    print("test accuracy: {}".format(accuracy))

    plt.plot(range(0, epochs), val_accuracies, c='b', label='val accuracy')
    plt.plot(range(0, epochs), train_accuracies, c='g', label='train accuracy')
    plt.legend(loc='upper left')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    # recog_demo(n, test)

def recog_demo(net, data):
    for iter in range(0, data[0].size):
        data_i = data[0][iter]
        predicted = net.predict(data_i)
        actual = data[1][iter]
        if predicted != actual:
            print("predicted {}, actual {}".format(predicted, actual))
            image = np.reshape(data_i, (28, 28))
            imgplot = plt.imshow(image, cmap=mpl.cm.Greys)
            imgplot.set_interpolation('nearest')
            plt.pause(10)
    # for iter in range(0, 25):
    #     data_i = data[0][iter]
    #     print(net.predict(data_i))
    #     image = np.reshape(data_i, (28, 28))
    #     imgplot = plt.imshow(image, cmap=mpl.cm.Greys)
    #     imgplot.set_interpolation('nearest')
    #     plt.pause(5)
if __name__ == '__main__':
    main()
