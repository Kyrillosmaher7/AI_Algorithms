
import numpy as np

from Lib.Losses.Loss import Loss


class CrossEntropyLoss(Loss):
    """
    Multi-Class Cross-Entropy Loss.

    Cross-Entropy Loss measures the difference between the true class
    distribution and the predicted probability distribution produced by
    a Softmax activation.

    This loss is used for multi-class classification problems where each
    sample belongs to exactly one class.

    The predicted values (`y_pred`) must be probability distributions
    (i.e., each row sums to 1), typically obtained by applying a Softmax
    activation to the network's logits.

    Loss Formula:
        L = -1/N * Σ Σ y_ij * log(p_ij)

    where:
        - N is the number of samples.
        - y_ij is the one-hot encoded ground-truth label.
        - p_ij is the predicted probability for class j.

    During backpropagation, this implementation computes the gradient
    of the loss with respect to the predicted probabilities:

        ∂L/∂p = -y / (p * N)

    If this loss is placed immediately after a Softmax layer, the Softmax
    backward pass combines with this gradient to produce the simplified
    gradient with respect to the logits:

        ∂L/∂z = (p - y) / N

    which is the gradient used by most deep learning frameworks.

    Notes
    -----
    - `y_pred` must contain probabilities, not raw logits.
    - `y_true` must be one-hot encoded.
    - Predictions are clipped to avoid log(0) and division by zero.
    - The returned loss is averaged over the batch.

    Parameters
    ----------
    y_true : np.ndarray
        One-hot encoded labels of shape (batch_size, num_classes).

    y_pred : np.ndarray
        Predicted probabilities after the Softmax activation with shape
        (batch_size, num_classes).

    Returns
    -------
    forward(...)
        float
            Average Cross-Entropy loss.

    backward(...)
        np.ndarray
            Gradient of the loss with respect to the predicted
            probabilities.
    """

    def __init__(self):
        super().__init__("Cross Entropy Loss")

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the average Cross-Entropy loss.

        Parameters
        ----------
        y_true : np.ndarray
            One-hot encoded ground-truth labels.

        y_pred : np.ndarray
            Predicted probabilities after Softmax.

        Returns
        -------
        float
            Average loss over the batch.
        """
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)

        loss = -np.sum(y_true * np.log(y_pred), axis=1)

        return np.mean(loss)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the predicted
        probabilities.

        Parameters
        ----------
        y_true : np.ndarray
            One-hot encoded labels.

        y_pred : np.ndarray
            Predicted probabilities after Softmax.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to y_pred.
        """

        batch_size = y_true.shape[0]

        return (y_pred - y_true) / batch_size