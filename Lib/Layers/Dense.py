import numpy as np

from Lib.Initializers import Initializer
from Lib.Layers.base import Layer
from Lib.Parameters.paramter import Parameter


class Dense(Layer):
    """
    Fully Connected (Dense) Layer.

    Computes:

        Y = XW + b

    where

        X : (batch_size, in_features)
        W : (in_features, out_features)
        b : (1, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        initializer:Initializer,
        use_bias: bool = True
    ):

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # Trainable parameters
        self.weight = Parameter(
            initializer((in_features, out_features))
        )

        self.bias = (
            Parameter(np.zeros((1, out_features)))
            if use_bias
            else None
        )

        # Cached tensors
        self._input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward propagation.

        Parameters
        ----------
        x : ndarray
            Shape (batch_size, in_features)

        Returns
        -------
        ndarray
            Shape (batch_size, out_features)
        """

        if x.ndim != 2:
            raise ValueError(
                f"Dense expects a 2D input, got {x.ndim}D."
            )

        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features, "
                f"got {x.shape[1]}."
            )

        self._input = x

        output = x @ self.weight.data

        if self.use_bias:
            output += self.bias.data

        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward propagation.

        Parameters
        ----------
        grad_output : ndarray
            Gradient received from the next layer.

        Returns
        -------
        ndarray
            Gradient with respect to the input.
        """

        # dL/dW
        self.weight.grad += self._input.T @ grad_output

        # dL/db
        if self.use_bias:
            self.bias.grad += np.sum(
                grad_output,
                axis=0,
                keepdims=True
            )

        # dL/dX
        grad_input = grad_output @ self.weight.data.T

        return grad_input

    def parameters(self):
        """
        Returns all trainable parameters.
        """

        if self.use_bias:
            return [self.weight, self.bias]

        return [self.weight]

    def __repr__(self):

        return (
            f"Dense("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"use_bias={self.use_bias})"
        )

    def get_config(self):

        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "use_bias": self.use_bias,
        }