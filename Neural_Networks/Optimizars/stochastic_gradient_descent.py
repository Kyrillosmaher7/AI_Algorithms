from .optim import Optimizer

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)
        self.params = None
        self.grads = None

    def step(self, params, grads):
        # Store the parameters for later use if needed
        self.params = params
        self.grads = grads
        
        # Update each parameter in place
        for i in range(len(params)):
            # Ensure gradient shape matches parameter shape
            if params[i].shape == grads[i].shape:
                params[i] -= self.lr * grads[i]
            else:
                raise ValueError(f"Shape mismatch: params[{i}] shape {params[i].shape} vs grads[{i}] shape {grads[i].shape}")