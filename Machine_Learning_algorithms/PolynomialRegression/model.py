
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from Neural_Networks.Model.model import Model
from Machine_Learning_algorithms.LinearRegression.model import LinearRegressionModel
class PolynomialRegressionModel(Model):
    """
    Polynomial Regression using Linear Regression with polynomial features
    
    Mathematical Formulation:
    y = w₀ + w₁x + w₂x² + w₃x³ + ... + wₙxⁿ
    
    Where:
    - x, x², x³, ... are polynomial features
    - w₀, w₁, w₂, ... are the coefficients
    - n is the degree of the polynomial
    """
    
    def __init__(self, degree=2, include_bias=True):
        """
        Initialize polynomial regression
        
        Parameters:
        -----------
        degree : int, default=2
            The degree of the polynomial (e.g., 2 for quadratic, 3 for cubic)
        include_bias : bool, default=True
            Whether to include a bias term (intercept)
        """
        self.degree = degree
        self.include_bias = include_bias
        self.weights = None
        self.bias = None
        self.feature_names = None
        
    def _create_polynomial_features(self, X):
        """
        Create polynomial features from input data
        
        If X has shape (n_samples, 1), then for degree=2:
        Original: [x₁, x₂, ..., xₙ]
        Transformed: [x₁, x₁², x₂, x₂², ..., xₙ, xₙ²]
        
        If X has multiple features, it creates interaction terms too:
        For degree=2 with features [x₁, x₂]:
        Transformed: [x₁, x₂, x₁², x₁x₂, x₂²]
        """
        n_samples = X.shape[0]
        
        # If X is 1D, reshape to 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        n_features = X.shape[1]
        
        # Start with original features
        poly_features = [X]
        
        # Add polynomial terms
        for d in range(2, self.degree + 1):
            # For each feature, add x^d
            poly_features.append(X ** d)
            
            # Add interaction terms if multiple features
            if n_features > 1 and d == 2:
                for i in range(n_features):
                    for j in range(i+1, n_features):
                        poly_features.append((X[:, i:i+1] * X[:, j:j+1]))
        
        # Combine all features
        X_poly = np.hstack(poly_features)
        
        # Create feature names for better interpretation
        self._create_feature_names(n_features)
        
        return X_poly
    
    def _create_feature_names(self, n_original_features):
        """Create readable feature names"""
        names = []
        for d in range(1, self.degree + 1):
            for i in range(n_original_features):
                names.append(f'x{i+1}^{d}')
        
        # Add interaction terms
        if n_original_features > 1:
            for i in range(n_original_features):
                for j in range(i+1, n_original_features):
                    names.append(f'x{i+1}*x{j+1}')
        
        self.feature_names = names
    
    def fit(self, X, y, optimizer=None, loss_fn=None, epochs=1000, 
            use_sklearn=True):
        """
        Fit polynomial regression model
        
        Two methods available:
        1. Using scikit-learn's LinearRegression (fast, stable)
        2. Using your custom gradient descent (educational)
        """
        
        # Create polynomial features
        X_poly = self._create_polynomial_features(X)
        
        if use_sklearn:
            # Use scikit-learn's linear regression (fast and stable)
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression(fit_intercept=self.include_bias)
            self.model.fit(X_poly, y)
            
            # Store parameters
            if self.include_bias:
                self.weights = self.model.coef_
                self.bias = self.model.intercept_
            else:
                self.weights = self.model.coef_
                self.bias = 0
        else:
            # Use your custom implementation (good for learning)
            # This assumes you have your LinearRegressionModel class
            
            self.model = LinearRegressionModel()
            self.model.fit(X_poly, y, optimizer, loss_fn, epochs)
            
            self.weights = self.model.weights
            self.bias = self.model.bias
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_poly = self._create_polynomial_features(X)
        
        
        return np.dot(X_poly, self.weights) + self.bias
    
    def get_coefficients(self):
        """Return the polynomial coefficients"""
        return {
            'coefficients': self.weights,
            'intercept': self.bias,
            'feature_names': self.feature_names
        }
