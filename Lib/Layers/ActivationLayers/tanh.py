import numpy as np
from Lib.Layers.ActivationLayers.base import Activation

class TanhLayer(Activation):

    def __init__(self):
        super().__init__("Tanh")

    def forward(self, x):

        self._output = np.tanh(x)

        return self._output

    def backward(self, grad_output):

        return grad_output * (1 - self._output ** 2)