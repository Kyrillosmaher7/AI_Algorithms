
import numpy as np
from Neural_Networks.Losses.Loss import Loss
from Neural_Networks.Optimizars.optim import Optimizer


class Model:
    """
    Base class for all models.
    """
    def __init__(self):
        pass
    def fit(self, X: np.ndarray, y: np.ndarray, optimizer: Optimizer, loss_fn: Loss, epochs=1000, batch_size=None):
        """
        Train the model on the given data.
        """
        raise NotImplementedError("The fit method must be implemented by subclasses.")
    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        raise NotImplementedError("The predict method must be implemented by subclasses.")