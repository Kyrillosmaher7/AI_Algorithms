import numpy as np
from .model import LinearRegressionModel as CustomLinearRegression
from sklearn.linear_model import LinearRegression
from Neural_Networks.Losses.mse import MSELoss
from Neural_Networks.Optimizars.stochastic_gradient_descent import SGD
from test import LinearRegressionModel
from sklearn.model_selection import train_test_split
import pandas as dp
from sklearn.preprocessing import StandardScaler

def test_linear_regression(X, y):
    """
    Test the custom linear regression implementation
    """
    # Initialize model, optimizer, and loss function
    model = CustomLinearRegression()
    optimizer = SGD(learning_rate=0.01)  # alpha = 0.01
    loss_fn = MSELoss()                  # MSE loss function

    # Train the model
    model.fit(X, y, optimizer, loss_fn, epochs=2000)

    # Make predictions
    predictions = model.predict(X)
    
    # Print final parameters
    print(f'Final weights: {model.weights}, Final bias: {model.bias}')
    return predictions

def test_linear_regression_scikit_learn(X, y):
    """
    Test scikit-learn's linear regression for comparison
    """
    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)
    print(f'Final weights: {model.coef_}, Final bias: {model.intercept_}')
    return predictions


def main():
    """
    Main function to test the implementation
    """
    
    # Read the dataset
    data = dp.read_csv("DataSets/fetch_california_housing.csv")
    X = data.drop("MedHouseVal", axis=1) # Features
    y = data["MedHouseVal"] # Target variable

    # Split the dataset into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("=" * 50)
    print("Testing custom Linear Regression Model:")
    print("=" * 50)
    test_linear_regression(X_train, y_train)

    print("\n" + "=" * 50)
    print("Testing scikit-learn Linear Regression Model:")
    print("=" * 50)
    test_linear_regression_scikit_learn(X_test, y_test)


if __name__ == "__main__":
    main()