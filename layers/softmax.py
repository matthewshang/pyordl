import numpy as np
from layer import Layer

class Softmax(Layer):
    def __init__(self):
        self.input_size = 0
        self.output_size = 0

    def forward(self, input):
        exp = np.exp(input)
        return exp / np.sum(exp, axis=1, keepdims=True)

    # gradient is labels
    def backward(self, input, output, gradient):
        new_grad = (output - gradient) / output.shape[0]
        return new_grad, 0.0

    def get_loss(self, output, labels):
        return -np.multiply(labels, np.log(output)).sum() / output.shape[0]
