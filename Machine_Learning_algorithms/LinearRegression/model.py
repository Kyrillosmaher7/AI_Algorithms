import numpy as np

from Lib.Model.model import Model
from Lib.Losses.Loss import Loss
from Lib.Losses.mse import MSELoss
from Lib.Optimizars.optim import Optimizer
from Lib.Optimizars.stochastic_gradient_descent import SGD
from Lib.Parameters.paramter import Parameter


class LinearRegressionModel(Model):
    """
    Linear Regression Model using Gradient Descent
    
    Mathematical Formulation:
    - Hypothesis: h(x) = W·X + b
    - Loss Function: J(W,b) = (1/n) * Σ(h(x_i) - y_i)²
    - Gradient Descent Update:
        W = W - alpha * ∂J/∂W
        b = b - alpha * ∂J/∂b
    where alpha is the learning rate
    """
    
    def __init__(self):

        self.weights = None  # Weight Parameter W
        self.bias = None     # Bias Parameter b

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
        Train the linear regression model using gradient descent
        
        Mathematical Steps:
        1. Initialize parameters (W, b)
        2. For each epoch:
            a. Forward pass: Compute predictions ŷ = X·W + b
            b. Compute loss: J = (1/n) * Σ(ŷ - y)²
            c. Backward pass: Compute gradients
               - ∂J/∂W = (2/n) * Xᵀ·(ŷ - y)
               - ∂J/∂b = (2/n) * Σ(ŷ - y)
            d. Update parameters: W = W - alpha·∂J/∂W, b = b - alpha·∂J/∂b

        Train Linear Regression using Batch / SGD / Mini-Batch Gradient Descent.

        GD type depends on batch_size:
        - batch_size = None → Batch Gradient Descent
        - batch_size = 1 → SGD
        - batch_size > 1 → Mini-Batch GD
        """

        # Step 1: Get dimensions of the data
        # X has shape (n_samples, n_features)

        n_samples, n_features = X.shape


        # Step 2: Initialize parameters
        # W: weight vector of size n_features
        # b: bias scalar

        self.weights = Parameter(
            np.zeros(n_features)
        )

        self.bias = Parameter(
            np.array(0.0)
        )


        # Ensure target vector y is 1D for proper broadcasting
        # y should have shape (n_samples,)

        if len(y.shape) > 1:
            y = y.flatten()



        # Step 3: Gradient Descent Loop

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

            # Hypothesis: ŷ = X·W + b

            y_pred = (
                np.dot(
                    X_batch,
                    self.weights.data
                )
                +
                self.bias.data
            )



            # ============================================
            # STEP 3b: COMPUTE LOSS
            # ============================================

            loss = loss_fn.forward(
                y_batch,
                y_pred
            )



            # ============================================
            # STEP 3c: BACKWARD PASS
            # ============================================


            # Compute gradient of loss w.r.t predictions

            dLoss = loss_fn.backward(
                y_batch,
                y_pred
            )


            # Compute gradient w.r.t weights

            self.weights.grad = np.dot(
                X_batch.T,
                dLoss
            )


            # Compute gradient w.r.t bias

            self.bias.grad = np.sum(
                dLoss
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
        
        Mathematical: ŷ = X·W + b
        """

        return (
            np.dot(
                X,
                self.weights.data
            )
            +
            self.bias.data
        )