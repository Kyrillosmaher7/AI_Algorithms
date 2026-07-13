from Lib.Initializers.Initializer import Initializer


import numpy as np

class RandomNormal(Initializer):

    def __init__(self, mean=0.0, std=0.01):
        self.mean = mean
        self.std = std

    def __call__(self, shape):
        return np.random.normal(
            self.mean,
            self.std,
            shape
        )