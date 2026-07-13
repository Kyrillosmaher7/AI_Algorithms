

import numpy as np

from Lib.Initializers.Initializer import Initializer


class Zeros(Initializer):
    def _call__(self, shape):
       return np.zeros(shape= shape)