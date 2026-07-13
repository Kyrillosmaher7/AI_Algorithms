import numpy as np

from Lib.Model.model import Model


class Sequential(Model):
    """
    Sequential neural network model.

    Executes layers in order:

        Input
          |
        Layer 1
          |
        Layer 2
          |
        Output

    Example:

        model = Sequential([
            Dense(2, 8),
            ReLU(),
            Dense(8, 1),
            Sigmoid()
        ])
    """

    def __init__(self, layers=None):

        self.layers = []

        if layers is not None:
            self.layers.extend(layers)

        self.optimizer = None
        self.loss_function = None

        self.history = {
            "loss": []
        }


    def add(self, layer):
        """
        Add a new layer to the model.
        """

        self.layers.append(layer)


    def forward(self, x):
        """
        Forward propagation through all layers.
        """

        for layer in self.layers:
            x = layer.forward(x)

        return x


    def backward(self, grad):
        """
        Backward propagation through all layers.
        """

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad


    def parameters(self):
        """
        Collect all trainable parameters from every layer.
        """

        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params

    def compile(
        self,
        optimizer,
        loss
    ):
        """
        Configure training components.

        Example:

            model.compile(
                optimizer=SGD(0.1),
                loss=BinaryCrossEntropy()
            )
        """

        self.optimizer = optimizer
        self.loss_function = loss


    def fit(
        self,
        X,
        y,
        epochs=100,
        verbose=True
    ):
        """
        Train the neural network.

        Training loop:

            Forward
              |
            Loss
              |
            Backward
              |
            Update weights
        """

        if self.optimizer is None:
            raise Exception(
                "Optimizer is not set. "
                "Call compile() first."
            )

        if self.loss_function is None:
            raise Exception(
                "Loss function is not set."
            )


        for epoch in range(epochs):

            # Forward pass
            predictions = self.forward(X)

            # Compute loss
            loss = self.loss_function.forward(
                y,
                predictions
            )

            # Compute dL/dPrediction
            grad = self.loss_function.backward(
                y,
                predictions
            )

            # Backpropagate through all layers
            self.backward(grad)
            self.optimizer.step(self.parameters())

            # Reset gradients
            self.optimizer.zero_grad(
                self.parameters()
            )

            # Save loss
            self.history["loss"].append(loss)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"- Loss: {loss:.6f}"
                )
        return self.history


    def predict(self, X):
        """
        Generate predictions.
        """

        return self.forward(X)



    def summary(self):
        """
        Print model architecture.
        """

        print("\nModel Summary")
        print("=" * 40)

        total_params = 0

        for i, layer in enumerate(self.layers):

            params = layer.parameters()

            layer_params = sum(
                p.data.size
                for p in params
            )

            total_params += layer_params


            print(
                f"{i}: {layer}"
                f"  Parameters: {layer_params}"
            )


        print("=" * 40)
        print(
            f"Total parameters: {total_params}"
        )