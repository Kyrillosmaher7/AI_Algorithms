import numpy as np
from Lib.Losses.Loss import Loss


class HuberLoss(Loss):
    """
    The Huber Loss Function is a popular loss function used primarily in regression tasks. 
    It is designed to be robust to outliers combining the best properties of two common loss functions: Mean Squared Error (MSE) and Mean Absolute Error (MAE).
    Unlike MSE, which can be heavily influenced by large errors (outliers) and MAE which can be less sensitive to small errors
    the Huber loss behaves like MSE for small prediction errors and switches to MAE for larger errors.
    This is useful when your dataset contains noisy data or outliers, helping models learn more reliably and avoid being skewed by extreme values.
    
    Formula:
        L(y_true, y_pred) =
                                0.5 * (y_true - y_pred)^2                     if |y_true - y_pred| <= delta
                                delta * |y_true - y_pred| - 0.5 * delta^2     otherwise
                                
    """
    def __init__(self, delta=1.0):
        super().__init__("Huber Loss")
        self.delta = delta

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        error = y_true - y_pred
        is_small_error = np.abs(error) <= self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * (np.abs(error) - 0.5 * self.delta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        error = y_true - y_pred
        is_small_error = np.abs(error) <= self.delta
        grad = np.where(is_small_error, -error, -self.delta * np.sign(error))
        return grad / y_true.size