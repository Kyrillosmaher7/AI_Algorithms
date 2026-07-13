from Lib.Model.model import Model
import numpy as np
from Lib.Losses.Loss import Loss
from Lib.Optimizars.optim import Optimizer

class GaussianNaiveBayesModel(Model):
    """
    Gaussian Naive Bayes classifier for continuous features.
    """
    
    def __init__(self):
        super().__init__()
        self.classes_prior = {}  # Changed from classes_parior
        self.feature_stats = {}
        self.classes = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, optimizer: Optimizer = None, 
            loss_fn: Loss = None, epochs: int = 1000, batch_size: int = None):
        """
        Fit Gaussian Naive Bayes model
        """
        # Get unique classes
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        # Calculate class priors
        for cls in self.classes:
            # Count samples in each class
            class_mask = (y == cls)
            class_count = np.sum(class_mask)
            
            # Calculate prior probability P(Y=cls)
            self.classes_prior[cls] = class_count / n_samples  # Fixed attribute name
            
            # Calculate mean and standard deviation for each feature in this class
            class_features = X[class_mask]
            self.feature_stats[cls] = {
                'mean': np.mean(class_features, axis=0),
                'std': np.std(class_features, axis=0) + 1e-9  # Add epsilon to avoid division by zero
            }
        
        return self
    
    def _gaussian_probability(self, x, mean, std):
        """
        Calculate Gaussian probability density function
        
        Parameters:
        -----------
        x : scalar or np.ndarray
            Feature value(s)
        mean : scalar or np.ndarray
            Mean of the feature
        std : scalar or np.ndarray
            Standard deviation of the feature
            
        Returns:
        --------
        probability : np.ndarray
            Gaussian probability
        """
        exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        denominator = std * np.sqrt(2 * np.pi)
        return exponent / denominator
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized version for better performance
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        # Ensure X is 2D
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        log_posteriors = np.zeros((n_samples, n_classes))
        
        for idx, cls in enumerate(self.classes):
            # Log prior
            log_prior = np.log(self.classes_prior[cls])
            
            # Get class statistics
            mean = self.feature_stats[cls]['mean']
            std = self.feature_stats[cls]['std']
            
            # Vectorized calculation of log likelihood for all samples
            # Shape: (n_samples, n_features)
            log_probs = -0.5 * np.log(2 * np.pi * std**2) - ((X - mean) ** 2) / (2 * std ** 2)
            
            # Sum across features for each sample
            log_likelihood = np.sum(log_probs, axis=1)
            
            # Log posterior
            log_posteriors[:, idx] = log_prior + log_likelihood
        
        # Convert to probabilities
        max_log = np.max(log_posteriors, axis=1, keepdims=True)
        posteriors = np.exp(log_posteriors - max_log)
        posteriors = posteriors / np.sum(posteriors, axis=1, keepdims=True)
        
        return posteriors
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X
        """
        posteriors = self._predict_proba(X)
        return self.classes[np.argmax(posteriors, axis=1)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate posterior probabilities for each class
        """
        return self._predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy of the model
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)