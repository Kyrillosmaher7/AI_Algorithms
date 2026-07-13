import numpy as np

from Lib.Layers.ActivationLayers.base import Activation


class SigmoidLayer(Activation):

    def __init__(self):
        super().__init__("Sigmoid")

    def forward(self, x):

        self._input = x

        self._output = 1.0 / (1.0 + np.exp(-x))

        return self._output

    def backward(self, grad_output):

        return grad_output * self._output * (1 - self._output)