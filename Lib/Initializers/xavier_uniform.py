import numpy as np

from Lib.Initializers.Initializer import Initializer

class XavierUniform(Initializer):

    def __call__(self, shape):

        fan_in, fan_out = shape

        limit = np.sqrt(6 / (fan_in + fan_out))

        return np.random.uniform(
            -limit,
            limit,
            shape
        )