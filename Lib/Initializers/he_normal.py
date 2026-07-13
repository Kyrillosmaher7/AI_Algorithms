import numpy as np

from Lib.Initializers.Initializer import Initializer


class HeNormal(Initializer):

    def __call__(self, shape):

        fan_in = shape[0]

        std = np.sqrt(2.0 / fan_in)

        return np.random.normal(
            0,
            std,
            shape
        )