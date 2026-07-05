import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .model import LinearRegressionModel as CustomLinearRegression
from Neural_Networks.Losses.mse import MSELoss
from Neural_Networks.Optimizars.stochastic_gradient_descent import SGD


def evaluate_model(name, y_true, y_pred):
    """Print regression metrics."""

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name}")
    print("-" * 40)
    print(f"MAE : {mae:.6f}")
    print(f"MSE : {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²  : {r2:.6f}")


def main():

    # ============================================
    # Load dataset
    # ============================================

    data = pd.read_csv("DataSets/fetch_california_housing.csv")

    X = data.drop("MedHouseVal", axis=1).values
    y = data["MedHouseVal"].values

    # ============================================
    # Train / Test split
    # ============================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # ============================================
    # Feature Scaling
    # ============================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ============================================
    # Custom Linear Regression
    # ============================================

    print("=" * 60)
    print("Training Custom Linear Regression")
    print("=" * 60)

    custom_model = CustomLinearRegression()

    custom_model.fit(
        X_train,
        y_train,
        optimizer=SGD(learning_rate=0.01),
        loss_fn=MSELoss(),
        epochs=2000,
        batch_size=None,      # Batch GD
    )

    custom_pred = custom_model.predict(X_test)

    print("\nCustom Parameters")
    print("-----------------")
    print("Weights:", custom_model.weights)
    print("Bias:", custom_model.bias)

    evaluate_model("Custom Model (Test Set)", y_test, custom_pred)

    # ============================================
    # Scikit-Learn Linear Regression
    # ============================================

    print("\n" + "=" * 60)
    print("Training Scikit-Learn Linear Regression")
    print("=" * 60)

    sklearn_model = LinearRegression()

    sklearn_model.fit(X_train, y_train)

    sklearn_pred = sklearn_model.predict(X_test)

    print("\nScikit-Learn Parameters")
    print("-----------------------")
    print("Weights:", sklearn_model.coef_)
    print("Bias:", sklearn_model.intercept_)

    evaluate_model("Scikit-Learn Model (Test Set)", y_test, sklearn_pred)


if __name__ == "__main__":
    main()