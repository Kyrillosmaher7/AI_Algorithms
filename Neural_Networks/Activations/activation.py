
class Activation:
    def __init__(self, name):
        self.name = name

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, x):
        raise NotImplementedError("Backward method not implemented.")