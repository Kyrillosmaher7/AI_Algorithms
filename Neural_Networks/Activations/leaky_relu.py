import numpy as np
from Neural_Networks.Activations.activation import Activation

class LeakyReLU(Activation):
    """
    Leaky ReLU activation function.
    Formula: f(x) = x if x > 0 else alpha * x
    Used to address the "dying ReLU" problem by allowing a small, non-zero gradient when the unit is not active.
    """

    def __init__(self, alpha=0.01):
        super().__init__("LeakyReLU")
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)