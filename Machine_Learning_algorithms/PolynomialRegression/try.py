

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from Machine_Learning_algorithms.PolynomialRegression.model import PolynomialRegressionModel
from Machine_Learning_algorithms.LinearRegression.model import LinearRegressionModel
from Neural_Networks.Losses.mse import MSELoss
from Neural_Networks.Optimizars.stochastic_gradient_descent import SGD



def generate_nonlinear_data(n_samples=100, noise=0.3):
    """Generate data from different non-linear functions"""
    np.random.seed(42)
    X = np.random.rand(n_samples, 1) * 10 - 5  # Range: [-5, 5]
    
    # Different non-linear functions
    functions = {
        'quadratic': 2 * X.flatten()**2 - 3 * X.flatten() + 4,
        'cubic': 0.5 * X.flatten()**3 - 2 * X.flatten()**2 + X.flatten() + 3,
        'sine': np.sin(2 * X.flatten()) + X.flatten(),
        'exponential': np.exp(0.5 * X.flatten()) - X.flatten(),
    }
    
    # Add noise to each function
    for key in functions:
        functions[key] += noise * np.random.randn(n_samples)
    
    return X, functions

def demo_polynomial_regression():
    """Demonstrate polynomial regression on various non-linear functions"""
    
    # Generate data
    X, functions = generate_nonlinear_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (func_name, y) in enumerate(functions.items()):
        ax = axes[idx]
        
        # Plot original data
        ax.scatter(X, y, alpha=0.6, label='Data', color='blue')
        
        # Try different polynomial degrees
        for degree in [1, 2, 3, 5]:
            model = PolynomialRegressionModel(degree=degree)
            model.fit(X, y, use_sklearn=True)
            
            # Predict for smooth curve
            X_test = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
            y_pred = model.predict(X_test)
            
            ax.plot(X_test, y_pred, label=f'Degree {degree}', 
                   linewidth=2, alpha=0.7)
        
        ax.set_title(f'{func_name.capitalize()} Function')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_degrees():
    """Compare how different polynomial degrees fit the data"""
    
    # Generate non-linear data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = 0.5 * X.flatten()**3 - 2 * X.flatten()**2 + X.flatten() + 2 + 0.5 * np.random.randn(100)
    
    # Test different degrees
    degrees = [1, 2, 3, 4, 5, 8]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, degree in enumerate(degrees):
        ax = axes[idx]
        
        # Fit polynomial
        model = PolynomialRegressionModel(degree=degree)
        model.fit(X, y, use_sklearn=True)
        
        # Predict
        X_test = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        y_pred_train = model.predict(X)
        mse = mean_squared_error(y, y_pred_train)
        r2 = r2_score(y, y_pred_train)
        
        # Plot
        ax.scatter(X, y, alpha=0.5, label='Data', color='blue')
        ax.plot(X_test, y_pred, 'r-', linewidth=2, label=f'Degree {degree}')
        
        ax.set_title(f'Degree {degree}\nMSE: {mse:.3f}, R²: {r2:.3f}')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Show coefficients
        coefs = model.get_coefficients()
        print(f"\nDegree {degree} coefficients:")
        for name, coef in zip(coefs['feature_names'][:5], coefs['coefficients'][:5]):
            print(f"  {name}: {coef:.3f}")
        print(f"  Intercept: {coefs['intercept']:.3f}")
        if len(coefs['coefficients']) > 5:
            print(f"  ... and {len(coefs['coefficients']) - 5} more terms")
    
    plt.tight_layout()
    plt.show()

def custom_gradient_descent_polynomial():
    """Use your custom gradient descent for polynomial regression"""
    
    print("\n" + "="*50)
    print("Polynomial Regression with Custom Gradient Descent")
    print("="*50)
    
    # Generate data
    np.random.seed(42)
    X = np.random.rand(100, 1) * 6 - 3
    y = 2 * X.flatten()**2 - 3 * X.flatten() + 4 + 0.3 * np.random.randn(100)
    
    # Create polynomial features
    poly = PolynomialRegressionModel(degree=2)
    X_poly = poly._create_polynomial_features(X)
    
    print(f"Original X shape: {X.shape}")
    print(f"Polynomial features shape: {X_poly.shape}")
    print(f"Feature names: {poly.feature_names}")


    
    model = LinearRegressionModel()
    optimizer = SGD(learning_rate=0.01)
    loss_fn = MSELoss()
    
    model.fit(X_poly, y, optimizer, loss_fn, epochs=1000)
    
    print(f"\nTrained coefficients:")
    for name, coef in zip(poly.feature_names, model.weights):
        print(f"  {name}: {coef:.4f}")
    print(f"  Intercept: {model.bias:.4f}")

if __name__ == "__main__":
    # 1. Visualize polynomial regression on different functions
    demo_polynomial_regression()
    
    # 2. Compare different polynomial degrees
    compare_degrees()
    
    # 3. Use your custom gradient descent
    custom_gradient_descent_polynomial()