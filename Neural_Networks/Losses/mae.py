import numpy as np
from Neural_Networks.Losses.Loss import Loss


class MAELoss(Loss):
    """
    Mean Absolute Error (MAE) Loss function.
    Formula: L(y_true, y_pred) = mean(|y_true - y_pred|)
    Used for regression problems to measure the average absolute difference between true and predicted values.
    Choose MAE when:
        - Your dataset contains many outliers.
        - Every error should contribute proportionally.
        - You want a metric that's easy to interpret.
        - Robustness is more important than aggressively correcting large errors.
    """
    def __init__(self):
        super().__init__("Mean Absolute Error Loss")

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.sign(y_pred - y_true) / y_true.size