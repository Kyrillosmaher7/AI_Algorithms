from abc import ABC, abstractmethod

from Lib.Layers.base import Layer

class Activation(Layer, ABC):

    def __init__(self, name: str):
        self.name = name
        self._input = None

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, grad_output):
        pass
    def __repr__(self):
        return f"{self.name}()"