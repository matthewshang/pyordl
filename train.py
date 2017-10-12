import numpy as np
from net import Net
from layers import Linear, Relu, Softmax
import spiral

def main():

    data, labels = spiral.load(100, 2, 3)
    one_hot = np.zeros((labels.shape[0], labels.max() + 1))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    
    reg = 1e-3
    step_size = 1e-0
    n = Net()
    n.add_layer(Linear(2, 100))
    n.add_layer(Relu())
    n.add_layer(Linear(100, 25))
    n.add_layer(Relu())
    n.add_layer(Linear(25, 3))
    n.add_layer(Softmax())

    for i in range(5000):
        out = n.forward(data)
        loss = n.backprop(one_hot)
        reg_loss = 0.5 * reg * loss
        data_loss = n.layers[-1].get_loss(out, one_hot)
        cost = reg_loss + data_loss        
        n.update(reg, step_size)

        if i % 100 == 0:
            print("iter {} cost: {}".format(i, cost))
    
    spiral.plot(data, labels, n)

if __name__ == '__main__':
    main()