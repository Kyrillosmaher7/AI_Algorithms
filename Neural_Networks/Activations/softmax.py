import numpy as np
from Neural_Networks.Activations.activation import Activation


class Softmax(Activation):
    """
    Softmax activation function.
    Formula: f(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    Used for multi-class classification problems and outputs values in the range (0, 1) that sum to 1.
    """
    def __init__(self):
        super().__init__("softmax")

    def forward(self, x:np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, x:np.ndarray) -> np.ndarray:
        s = self.forward(x)
        return s * (1 - s)