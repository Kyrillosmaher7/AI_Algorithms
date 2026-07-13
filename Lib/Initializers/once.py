import numpy as np

from Lib.Initializers.Initializer import Initializer


class OnesInitializer(Initializer):
    def __int__(self):
        pass
    def __call__(self, shape):
        return np.ones(
            shape
        )