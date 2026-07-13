import numpy as np

from Lib.Model.model import Model
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


    def __init__(
        self,
        degree=2,
        include_bias=True
    ):

        """
        Initialize polynomial regression
        
        Parameters:
        -----------
        degree : int
            Degree of polynomial features.

        include_bias : bool
            Whether to include intercept term.
        """


        self.degree = degree

        self.include_bias = include_bias


        # Internal linear regression model

        self.model = None


        # Polynomial feature names

        self.feature_names = None



    def _create_polynomial_features(self, X):

        """
        Create polynomial features from input data

        Example:

        X = [x]

        degree=3:

        [x, x², x³]
        """


        # If X is 1D, reshape to 2D

        if len(X.shape) == 1:

            X = X.reshape(-1,1)



        n_features = X.shape[1]


        poly_features = [X]


        # Add polynomial powers

        for d in range(2, self.degree + 1):

            poly_features.append(
                X ** d
            )


            # Add interaction terms

            if n_features > 1 and d == 2:

                for i in range(n_features):

                    for j in range(i + 1, n_features):

                        poly_features.append(
                            X[:,i:i+1] *
                            X[:,j:j+1]
                        )


        X_poly = np.hstack(
            poly_features
        )


        self._create_feature_names(
            n_features
        )


        return X_poly




    def _create_feature_names(
        self,
        n_original_features
    ):

        """
        Create readable feature names
        """


        names = []


        for d in range(1, self.degree + 1):

            for i in range(n_original_features):

                names.append(
                    f"x{i+1}^{d}"
                )


        if n_original_features > 1:

            for i in range(n_original_features):

                for j in range(i+1,n_original_features):

                    names.append(
                        f"x{i+1}*x{j+1}"
                    )


        self.feature_names = names





    def fit(
        self,
        X,
        y,
        optimizer=None,
        loss_fn=None,
        epochs=1000
    ):

        """
        Train Polynomial Regression

        Uses the custom LinearRegressionModel
        with polynomial transformed features.
        """


        # Create polynomial features

        X_poly = self._create_polynomial_features(
            X
        )
        # Normalize polynomial features
        self.mean = np.mean(X_poly, axis=0)
        self.std = np.std(X_poly, axis=0)

        self.std[self.std == 0] = 1

        X_poly = (
            X_poly - self.mean
        ) / self.std



        # Create linear regression model

        self.model = LinearRegressionModel()



        # Train on polynomial features

        self.model.fit(
            X_poly,
            y,
            optimizer,
            loss_fn,
            epochs
        )


        return self





    def predict(self, X):

        """
        Predict using:

        Polynomial Features
              |
              |
        Linear Regression

        y = X_poly.W + b
        """

        X_poly = self._create_polynomial_features(X)

        X_poly = (
            X_poly - self.mean
        ) / self.std


        return self.model.predict(
            X_poly
        )




    def parameters(self):

        """
        Return trainable parameters.

        Allows PolynomialRegressionModel
        to work with the same optimizer system.
        """


        if self.model is None:

            return []


        return [
            self.model.weights,
            self.model.bias
        ]





    def get_coefficients(self):

        """
        Return polynomial coefficients.
        """


        if self.model is None:

            return None


        return {

            "coefficients":
                self.model.weights.data,


            "intercept":
                self.model.bias.data,


            "feature_names":
                self.feature_names
        }