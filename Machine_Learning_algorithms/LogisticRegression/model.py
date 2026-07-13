import numpy as np

from Lib.Losses.Loss import Loss
from Lib.Optimizars.optim import Optimizer
from Lib.Model.model import Model
from Lib.Activations.sigmoid import Sigmoid
from Lib.Parameters.paramter import Parameter


class LogisticRegressionModel(Model):
    """
    Logistic Regression Model using Gradient Descent
    
    Mathematical Formulation:
    - Hypothesis: h(x) = sigmoid(W·X + b)
    - Loss Function: J(W,b) = -(1/n) * Σ(y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i))
    - Gradient Descent Update:
        W = W - alpha * ∂J/∂W
        b = b - alpha * ∂J/∂b
    where alpha is the learning rate
    """

    def __init__(self):

        self.weights = None  # Weight Parameter W
        self.bias = None     # Bias Parameter b

        self.loss_history = []  # To store loss values during training

        # Sigmoid activation function
        self._sigmoid = Sigmoid()


    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        optimizer: Optimizer,
        loss_fn: Loss,
        epochs=1000,
        batch_size=None
    ):

        """
        Train the logistic regression model using gradient descent
        
        Mathematical Steps:
        1. Initialize parameters (W, b)
        2. For each epoch:
            a. Forward pass: Compute predictions ŷ = sigmoid(X·W + b)
            b. Compute loss: J = -(1/n) * Σ(y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i))
            c. Backward pass: Compute gradients
               - ∂J/∂W = (1/n) * Xᵀ·(ŷ - y)
               - ∂J/∂b = (1/n) * Σ(ŷ - y)
            d. Update parameters:
               W = W - alpha·∂J/∂W
               b = b - alpha·∂J/∂b

        Train Logistic Regression using Batch / SGD / Mini-Batch Gradient Descent.

        GD type depends on batch_size:
        - batch_size = None → Batch Gradient Descent
        - batch_size = 1 → SGD
        - batch_size > 1 → Mini-Batch GD

        Parameters:
        - X: Input features, shape (n_samples, n_features)
        - y: Target values, shape (n_samples,)
        - optimizer: Optimizer instance (e.g., SGD)
        - loss_fn: Loss function instance
        - epochs: Number of training iterations
        - batch_size: Size of mini-batches for gradient descent
        """


        # Step 1: Get dimensions of the data
        # X has shape (n_samples, n_features)

        n_samples, n_features = X.shape



        # Step 2: Initialize parameters
        #
        # W: Weight vector of size n_features
        # b: Bias term
        #
        # Parameters are wrapped inside Parameter objects
        # to work with the optimizer system.

        self.weights = Parameter(
            np.zeros((n_features, 1))
        )

        self.bias = Parameter(
            np.zeros((1,))
        )



        # Ensure target vector y has correct shape
        # y should have shape (n_samples,)

        if len(y.shape) > 1:
            y = y.flatten()



        # Step 3: Training loop

        for epoch in range(epochs):


            # ====================================================
            # SELECT BATCH (THIS defines GD type)
            # ====================================================


            if batch_size is None:

                # Batch Gradient Descent → full dataset

                X_batch = X
                y_batch = y



            elif batch_size == 1:

                # Stochastic Gradient Descent → one sample

                idx = np.random.randint(n_samples)

                X_batch = X[idx:idx + 1]

                y_batch = y[idx:idx + 1]



            else:

                # Mini-Batch Gradient Descent

                idx = np.random.choice(
                    n_samples,
                    batch_size,
                    replace=False
                )

                X_batch = X[idx]

                y_batch = y[idx]



            # ============================================
            # STEP 3a: FORWARD PASS
            # ============================================

            # Linear model:
            #
            # z = X·W + b
            #
            # Then apply sigmoid:
            #
            # ŷ = sigmoid(z)

            linear_model = (
                np.dot(
                    X_batch,
                    self.weights.data
                )
                +
                self.bias.data
            )


            y_predicted = self._sigmoid.forward(
                linear_model
            )



            # ============================================
            # STEP 3b: COMPUTE LOSS
            # ============================================

            loss = loss_fn.forward(
                y_batch.reshape(-1, 1),
                y_predicted
            )


            # Store loss history

            self.loss_history.append(loss)



            # ============================================
            # STEP 3c: BACKWARD PASS
            # ============================================


            # Convert labels and predictions
            # into column vectors

            y_batch = y_batch.reshape(-1, 1)

            y_predicted = y_predicted.reshape(-1, 1)



            batch_size_current = X_batch.shape[0]



            # Gradient with respect to weights:
            #
            # ∂J/∂W = (1/n) Xᵀ(ŷ - y)

            self.weights.grad = (
                (1 / batch_size_current)
                *
                np.dot(
                    X_batch.T,
                    (y_predicted - y_batch)
                )
            )



            # Gradient with respect to bias:
            #
            # ∂J/∂b = (1/n) Σ(ŷ - y)

            self.bias.grad = (
                np.sum(
                    y_predicted - y_batch
                )
                /
                batch_size_current
            )



            # ============================================
            # STEP 3d: UPDATE PARAMETERS
            # ============================================


            optimizer.step(
                [
                    self.weights,
                    self.bias
                ]
            )


            # Clear gradients after update

            optimizer.zero_grad(
                [
                    self.weights,
                    self.bias
                ]
            )



            # Print progress every 100 epochs

            if epoch % 100 == 0:

                print(
                    f"Epoch {epoch}, Loss: {loss}"
                )



    def predict(self, X):

        """
        Make predictions using the trained model
        
        Mathematical:
            ŷ = sigmoid(X·W + b)

        where W and b are the trained parameters.
        """


        linear_model = (
            np.dot(
                X,
                self.weights.data
            )
            +
            self.bias.data
        )


        y_predicted = self._sigmoid.forward(
            linear_model
        )


        return np.round(
            y_predicted
        )