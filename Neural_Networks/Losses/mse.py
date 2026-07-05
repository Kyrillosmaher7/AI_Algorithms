import numpy as np
from Neural_Networks.Losses.Loss import Loss


class MSELoss(Loss):
    """
    Mean Squared Error (MSE) Loss function.
    Formula: L(y_true, y_pred) = mean((y_true - y_pred)^2)
    Used for regression problems to measure the average squared difference between true and predicted values.
    Choose MSE when:
        - Large prediction errors are especially costly.
        - You want smoother gradients for optimization.
        - You're training neural networks or most regression models.
    """
    def __init__(self):
        super().__init__("Mean Squared Error Loss")

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # dloss/dy_pred = (-2/n) * (y_true - y_pred) which is equivilant to (2/n) * (y_pred - y_true)
        return 2 * (y_pred - y_true) / y_true.size