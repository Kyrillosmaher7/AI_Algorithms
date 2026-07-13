import numpy as np
from Lib.Activations.activation import Activation

class Tanh(Activation):
    """
    Tanh activation function.
    Formula: f(x) = (2 / (1 + exp(-2x))) - 1
    Used for introducing non-linearity in neural networks.
    """
    def __init__(self):
        super().__init__("tanh")

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)