from .optim import Optimizer

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
        self.params = None
        self.grads = None

    def step(self, params):

        for p in params:
            p.data -= self.lr * p.grad