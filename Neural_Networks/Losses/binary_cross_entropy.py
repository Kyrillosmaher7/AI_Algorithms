import numpy as np
from Neural_Networks.Losses.Loss import Loss

class BinaryCrossEntropyLoss(Loss):
    """
    Binary Cross-Entropy (BCE) Loss.

    Binary Cross-Entropy measures the difference between the true binary labels
    and the predicted probabilities produced by a binary classifier.

    This loss is commonly used for:
        - Binary classification (two classes)
        - Multi-label classification (one independent sigmoid output per class)

    The predicted values (`y_pred`) must be probabilities in the range (0, 1),
    typically obtained by applying a Sigmoid activation function to the model's
    output logits.

    Loss Formula:
        L = -1/N * Σ [ y * log(p) + (1 - y) * log(1 - p) ]

    where:
        - y is the ground-truth label (0 or 1)
        - p is the predicted probability
        - N is the total number of samples

    During backpropagation, this implementation computes the gradient of the
    loss with respect to the predicted probabilities:

        ∂L/∂p = (p - y) / (p * (1 - p) * N)

    If this loss is used immediately after a Sigmoid activation, the Sigmoid
    layer will further compute:

        ∂p/∂z = p * (1 - p)

    resulting in the simplified gradient with respect to the logits:

        ∂L/∂z = p - y

    which is the same gradient used by numerically stable implementations such
    as BCEWithLogitsLoss.

    Notes
    -----
    - `y_pred` must contain probabilities, not raw logits.
    - Predictions are clipped to the interval [1e-15, 1 - 1e-15] to prevent
      numerical issues such as log(0) and division by zero.
    - The returned loss is the mean loss over all samples.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels with values in {0, 1}.

    y_pred : np.ndarray
        Predicted probabilities after the Sigmoid activation.

    Returns
    -------
    forward(...)
        float
            The average Binary Cross-Entropy loss.

    backward(...)
        np.ndarray
            Gradient of the loss with respect to the predicted probabilities.
    """


    def __init__(self):
        super().__init__("Binary Cross Entropy Loss")

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Clip predictions to avoid log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Clip predictions to avoid division by zero
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.size