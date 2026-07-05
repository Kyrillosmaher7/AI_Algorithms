import numpy as np
from Neural_Networks.Losses.Loss import Loss
from Neural_Networks.Losses.mse import MSELoss
from Neural_Networks.Optimizars.optim import Optimizer
from Neural_Networks.Optimizars.stochastic_gradient_descent import SGD
class LinearRegressionModel:
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
        self.weights = None  # Weight vector W
        self.bias = None     # Bias term b

    def fit(self, X: np.ndarray, y: np.ndarray, optimizer: Optimizer, loss_fn: Loss, epochs=1000, batch_size=None):
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

        Parameters:
        - X: Input features, shape (n_samples, n_features)
        - y: Target values, shape (n_samples,)
        - optimizer: Optimizer instance (e.g., SGD)
        - loss_fn: Loss function instance (e.g., MSELoss)
        - epochs: Number of training iterations
        - batch_size: Size of mini-batches for gradient descent
        """
        
        # Step 1: Get dimensions of the data
        # X has shape (n_samples, n_features)
        n_samples, n_features = X.shape
        
        # Step 2: Initialize parameters
        # W: weight vector of size n_features (start with zeros)
        # b: bias scalar (start with 0)
        self.weights = np.zeros(n_features)  # W = [0, 0, ..., 0]
        self.bias = np.array(0.0)            # b = 0

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
                idx = np.random.choice(n_samples, batch_size, replace=False)
                X_batch = X[idx]
                y_batch = y[idx]
            
            # ============================================
            # STEP 3a: FORWARD PASS
            # ============================================
            # Hypothesis: ŷ = X·W + b
            # Mathematical: h(x) = Σ(W_j * x_j) + b
            # Vectorized: ŷ = X @ W + b
            y_pred = np.dot(X_batch, self.weights) + self.bias  # Shape: (n_samples,)
            
            # ============================================
            # STEP 3b: COMPUTE LOSS
            # ============================================
            # Mean Squared Error (MSE) Loss:
            # J(W,b) = (1/n) * Σ(ŷ_i - y_i)²
            # where n = number of samples
            loss = loss_fn.forward(y_batch, y_pred)  # Scalar value
            
            # ============================================
            # STEP 3c: BACKWARD PASS (Compute Gradients)
            # ============================================
            
            # Compute gradient of loss w.r.t predictions
            # ∂J/∂ŷ = (2/n) * (ŷ - y)
            # This is the derivative of MSE with respect to predictions
            dLoss = loss_fn.backward(y_batch, y_pred)  # Shape: (n_samples,)
            
            # Compute gradient w.r.t weights using chain rule:
            # ∂J/∂W = ∂J/∂ŷ * ∂ŷ/∂W
            # ∂ŷ/∂W = X (since ŷ = X·W + b)
            # Therefore: ∂J/∂W = Xᵀ · ∂J/∂ŷ
            # Mathematical: ∂J/∂W_j = Σ(X_ij * (ŷ_i - y_i)) * (2/n)
            dLoss_dW = np.dot(X_batch.T, dLoss)  # Shape: (n_features,)
            
            # Ensure dLoss_dW is 1D (flatten if it's 2D)
            if len(dLoss_dW.shape) > 1:
                dLoss_dW = dLoss_dW.flatten()
            
            # Compute gradient w.r.t bias using chain rule:
            # ∂J/∂b = ∂J/∂ŷ * ∂ŷ/∂b
            # ∂ŷ/∂b = 1 (since ŷ = X·W + b)
            # Therefore: ∂J/∂b = Σ(∂J/∂ŷ) = Σ(2/n * (ŷ - y))
            dLoss_db = np.sum(dLoss)  # Scalar value
            
            # Convert dLoss_db to numpy array with shape (1,) for consistency
            if not isinstance(dLoss_db, np.ndarray):
                dLoss_db = np.array(dLoss_db)
            elif len(dLoss_db.shape) == 0:
                dLoss_db = dLoss_db.reshape(1)
            
            # ============================================
            # STEP 3d: UPDATE PARAMETERS (Gradient Descent)
            # ============================================
            # Gradient Descent Update Rule:
            # W_new = W_old - α * ∂J/∂W
            # b_new = b_old - α * ∂J/∂b
            # where α is the learning rate
            optimizer.step([self.weights, self.bias], [dLoss_dW, dLoss_db])
            
            # Print progress every 100 epochs
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')