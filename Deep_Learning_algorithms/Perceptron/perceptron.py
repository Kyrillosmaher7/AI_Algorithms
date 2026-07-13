import numpy as np

class Perceptron: 

    """
     The Perceptron is the first neural network algorithm ever proposed (Frank Rosenblatt, 1957).
     It is the simplest form of a neural network—a single artificial neuron that can perform binary classification.

     A perceptron receives several input values, multiplies each by a weight, adds a bias, and decides between two classes.
           x1 ----(w1)--\
                         \
           x2 ----(w2)----> Σ ----> Step ----> Output
                          /
           x3 ----(w3)-- /
                           +
                            bias
    Step Activation Function
       =>The perceptron then converts the number into a prediction.
    The rule is :
       if z ≥ 0
            output = 1
        else
            output = 0
    """
    
    def __init__(self, learning_rate: float = 0.01, epochs: int = 100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
    
    def step(self, z: float):
        return np.where(z >= 0, 1, 0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        

        self.w = np.zeros(n_features)
        self.b = 0 
        
        for _ in range(self.epochs):
            z = np.dot(X, self.w) + self.b 
            predictions = self.step(z)
            
            # Calculate errors
            error = y - predictions
            
            # Update weights and bias
            self.w += self.learning_rate * np.dot(error, X)  
            self.b += self.learning_rate * np.sum(error) 

    def predict(self, X):
        z = np.dot(X, self.w) + self.b
        return self.step(z)