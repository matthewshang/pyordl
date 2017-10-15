class Layer:
    def forward(self, input):
        pass

    def backward(self, input, output, gradient):
        pass

    def update(self, step_size):
        pass
