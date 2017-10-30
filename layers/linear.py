import numpy as np
from layer import Layer

class Linear(Layer):
    def __init__(self, output_size, input_size=0, reg=1e-3):
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.reg = reg
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, input):
        if self.W is None:
            self.W = 0.01 * np.random.randn(self.input_size, self.output_size)
        if self.b is None:
            self.b = np.zeros((1, self.output_size))
        return np.dot(input, self.W) + self.b

    def backward(self, input, output, gradient):
        self.dW = np.dot(input.T, gradient)
        self.db = np.sum(gradient, axis=0, keepdims=True)
        new_grad = np.dot(gradient, self.W.T)
        return new_grad, 0.5 * self.reg * np.sum(self.W * self.W)

    def update(self, step_size):
        self.dW += self.reg * self.W
        self.W += -step_size * self.dW
        self.b += -step_size * self.db
