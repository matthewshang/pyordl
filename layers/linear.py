import numpy as np
from layer import Layer

class Linear(Layer):
    def __init__(self, input_size, output_size):
        self.W = 0.01 * np.random.randn(input_size, output_size)
        self.b = np.zeros((1, output_size))
        self.dW = None
        self.db = None

    def forward(self, input):
        return np.dot(input, self.W) + self.b

    def backward(self, input, output, gradient):
        self.dW = np.dot(input.T, gradient)
        self.db = np.sum(gradient, axis=0, keepdims=True)
        new_grad = np.dot(gradient, self.W.T)
        return new_grad, np.sum(self.W * self.W)

    def update(self, reg, step_size):
        self.dW += reg * self.W
        self.W += -step_size * self.dW
        self.b += -step_size * self.db