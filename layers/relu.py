import numpy as np
from layer import Layer

class Relu(Layer):
    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, input, output, gradient):
        new_grad = np.empty_like(gradient)
        new_grad[:] = gradient
        new_grad *= (output > 0)
        return new_grad, 0.0