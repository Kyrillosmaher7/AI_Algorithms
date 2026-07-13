import numpy as np

from Lib.Activations.activation import Activation


class ReLU(Activation):
    """
    Rectified Linear Unit activation function.
    Formula: f(x) = max(0, x)
    Used in hidden layers of neural networks to introduce non-linearity.
    """
    def __init__(self):
        super().__init__("ReLU")

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input
    