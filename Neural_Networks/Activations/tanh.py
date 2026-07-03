import numpy as np
from Neural_Networks.Activations.activation import Activation

class Tanh(Activation):
    """
    Tanh activation function.
    Formula: f(x) = (2 / (1 + exp(-2x))) - 1
    Used for introducing non-linearity in neural networks.
    """
    def __init__(self):
        super().__init__("tanh")

    def forward(self, x:float) -> float:
        return (2 / (1 + np.exp(-2 * x))) - 1

    def backward(self, x:float) -> float:
        return 1 - np.tanh(x) ** 2