

from Neural_Networks.Activations.activation import Activation


class ReLU(Activation):
    """
    Rectified Linear Unit activation function.
    Formula: f(x) = max(0, x)
    Used in hidden layers of neural networks to introduce non-linearity.
    """
    def __init__(self):
        super().__init__("ReLU")

    def forward(self, x: float) -> float:
        return max(0, x)

    def backward(self, x: float) -> float:
        return 1 if x > 0 else 0
    