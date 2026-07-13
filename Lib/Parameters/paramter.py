import numpy as np

class Parameter:
    """
    Represents a trainable parameter.
    """

    def __init__(self, data: np.ndarray):

        self.data = data
        self.grad = np.zeros_like(data)

    def zero_grad(self):

        self.grad.fill(0.0)