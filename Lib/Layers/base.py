from Lib.Initializers import Initializer


class Layer:
    def __init__(
        self,
        in_features: int,
        out_features: int,
        initializer:Initializer,
        use_bias: bool = True
    ):pass
    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        return []