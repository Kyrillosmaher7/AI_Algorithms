import numpy as np

class Loss:
    def __init__(self, name):
        self.name = name

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Backward method not implemented.")