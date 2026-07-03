import numpy as np
from Neural_Networks.Activations.activation import Activation

class Sigmoid(Activation):
    """
    Sigmoid activation function.
    Formula: f(x) = 1 / (1 + exp(-x))
    Used for binary classification problems and outputs values in the range (0, 1).
    """
    def __init__(self):
        super().__init__("sigmoid")

    def forward(self, x:float) -> float:
        return 1 / (1 + np.exp(-x))

    def backward(self, x:float) -> float:
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1 - sigmoid_x)