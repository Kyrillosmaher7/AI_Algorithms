
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

from Machine_Learning_algorithms.PolynomialRegression.model import PolynomialRegressionModel
from Lib.Losses.mse import MSELoss
from Lib.Optimizars.stochastic_gradient_descent import SGD



def generate_nonlinear_data(
    n_samples=100,
    noise=0.3
):

    """
    Generate nonlinear regression data.

    Example:
        y = 2x² - 3x + 4 + noise
    """


    np.random.seed(42)


    X = (
        np.random.rand(
            n_samples,
            1
        )
        *
        10
        -
        5
    )


    y = (
        2 * X.flatten()**2
        -
        3 * X.flatten()
        +
        4
    )


    y += noise * np.random.randn(
        n_samples
    )


    return X, y




def test_polynomial_regression():


    print("="*60)
    print("Polynomial Regression Test")
    print("="*60)



    # ============================================
    # Generate Dataset
    # ============================================

    X, y = generate_nonlinear_data()



    # ============================================
    # Create Model Components
    # ============================================


    model = PolynomialRegressionModel(
        degree=2
    )


    optimizer = SGD(
        learning_rate=0.01
    )


    loss_fn = MSELoss()



    # ============================================
    # Train Model
    # ============================================


    model.fit(
        X,
        y,
        optimizer,
        loss_fn,
        epochs=2000
    )



    # ============================================
    # Predictions
    # ============================================


    predictions = model.predict(
        X
    )



    # ============================================
    # Metrics
    # ============================================


    mse = mean_squared_error(
        y,
        predictions
    )


    r2 = r2_score(
        y,
        predictions
    )


    print("\nResults")
    print("-"*40)

    print(
        f"MSE: {mse:.6f}"
    )


    print(
        f"R2 Score: {r2:.6f}"
    )



    # ============================================
    # Coefficients
    # ============================================


    coefficients = model.get_coefficients()


    print("\nCoefficients")
    print("-"*40)


    for name, coef in zip(
        coefficients["feature_names"],
        coefficients["coefficients"]
    ):

        print(
            f"{name}: {coef:.4f}"
        )


    print(
        f"Bias: {coefficients['intercept']}"
    )



    # ============================================
    # Plot Result
    # ============================================


    X_test = np.linspace(
        X.min(),
        X.max(),
        200
    ).reshape(-1,1)



    y_test = model.predict(
        X_test
    )


    plt.scatter(
        X,
        y,
        label="Data"
    )


    plt.plot(
        X_test,
        y_test,
        label="Polynomial Fit"
    )


    plt.title(
        "Polynomial Regression Degree 2"
    )


    plt.xlabel(
        "X"
    )


    plt.ylabel(
        "y"
    )


    plt.legend()

    plt.grid()

    plt.show()





def compare_polynomial_degrees():


    print("\n")
    print("="*60)
    print("Comparing Polynomial Degrees")
    print("="*60)



    X, y = generate_nonlinear_data()



    degrees = [
        1,
        2,
        3,
        5
    ]



    for degree in degrees:


        model = PolynomialRegressionModel(
            degree=degree
        )


        optimizer = SGD(
            learning_rate=0.01
        )


        loss_fn = MSELoss()



        model.fit(
            X,
            y,
            optimizer,
            loss_fn,
            epochs=2000
        )



        pred = model.predict(
            X
        )



        mse = mean_squared_error(
            y,
            pred
        )


        r2 = r2_score(
            y,
            pred
        )



        print(
            f"Degree {degree}"
        )

        print(
            f"MSE : {mse:.5f}"
        )


        print(
            f"R2  : {r2:.5f}"
        )

        print("-"*30)





if __name__ == "__main__":


    test_polynomial_regression()


    compare_polynomial_degrees()