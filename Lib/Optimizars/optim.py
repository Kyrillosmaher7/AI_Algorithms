import numpy as np


class Optimizer:
    """
    Base class for all optimizers.
    """

    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def step(self, params):
        """
        Update parameters.

        params: list of numpy arrays (weights, biases, etc.)
        grads: list of gradients matching params
        """
        raise NotImplementedError

    def zero_grad(self, params):
        """
        Optional helper if you store gradients.
        """
        for p in params:
            p.zero_grad()
