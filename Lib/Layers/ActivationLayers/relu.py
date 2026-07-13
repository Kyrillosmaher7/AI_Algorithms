import numpy as np

from Lib.Layers.ActivationLayers.base import Activation


class ReLULayer(Activation):

    def __init__(self):
        super().__init__("ReLU")

    def forward(self, x):

        self._input = x

        return np.maximum(0, x)

    def backward(self, grad_output):

        grad = grad_output.copy()

        grad[self._input <= 0] = 0

        return grad