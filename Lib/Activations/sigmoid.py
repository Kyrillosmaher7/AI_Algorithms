import numpy as np
from Lib.Activations.activation import Activation

class Sigmoid(Activation):
    """
    Sigmoid activation function.
    Formula: f(x) = 1 / (1 + exp(-x))
    Used for binary classification problems and outputs values in the range (0, 1).
    """
    def __init__(self):
        super().__init__("sigmoid")

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)
    