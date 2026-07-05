
import numpy as np
from Neural_Networks.Losses.Loss import Loss
from Neural_Networks.Optimizars.optim import Optimizer
from Neural_Networks.Model.model import Model
from Neural_Networks.Activations.sigmoid import Sigmoid

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
        self.weights = None  # Weight vector W
        self.bias = None     # Bias term b
        self.loss_history = []  # To store loss values during training
        self._sigmoid =  Sigmoid().forward  # Sigmoid activation function

    def fit(self, X: np.ndarray, y: np.ndarray, optimizer: Optimizer, loss_fn: Loss, epochs=1000, batch_size=None):
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
            d. Update parameters: W = W - alpha·∂J/∂W, b = b - alpha·∂J/∂b

        Train Logistic Regression using Batch / SGD / Mini-Batch Gradient Descent.

        GD type depends on batch_size:
        - batch_size = None → Batch Gradient Descent
        - batch_size = 1 → SGD
        - batch_size > 1 → Mini-Batch GD

        Parameters:
        - X: Input features, shape (n_samples, n_features)
        - y: Target values, shape (n_samples,)
        - optimizer: Optimizer instance (e.g., SGD)
        - loss_fn: Loss function instance (e.g., CrossEntropyLoss)
        - epochs: Number of training iterations
        - batch_size: Size of mini-batches for gradient descent
        """
        # Step 1: Get dimensions of the data
        n_samples, n_features = X.shape
        
        # Step 2: Initialize parameters
        self.weights = np.zeros((n_features, 1)) # Weight vector W
        self.bias = np.zeros((1,))  # Bias term b
        
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
                idx = np.random.choice(n_samples, batch_size, replace=False)
                X_batch = X[idx]
                y_batch = y[idx]
            
            # Forward pass: Compute predictions
            linear_model = np.dot(X_batch, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            
            # Compute loss
            loss = loss_fn.forward(y_batch, y_predicted)
            
            # Backward pass: Compute gradients
            y_batch = y_batch.reshape(-1, 1)
            y_predicted = y_predicted.reshape(-1, 1)

            batch_size = X_batch.shape[0]

            dw = (1 / batch_size) * np.dot(X_batch.T, (y_predicted - y_batch))
            db = np.sum(y_predicted - y_batch).reshape(1,)
            # Update parameters using optimizer
            optimizer.step([self.weights, self.bias], [dw, db])
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
            """
            Make predictions using the trained model
            Mathematical: ŷ = sigmoid(X·W + b)
            where W and b are the trained parameters
            """
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
            return np.round(y_predicted)