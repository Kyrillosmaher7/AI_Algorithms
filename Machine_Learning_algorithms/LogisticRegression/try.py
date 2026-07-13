import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Machine_Learning_algorithms.LogisticRegression.model import (
    LogisticRegressionModel as CustomLogisticRegression,
)
from Lib.Losses.binary_cross_entropy import BinaryCrossEntropyLoss
from Lib.Optimizars.stochastic_gradient_descent import SGD


def evaluate_model(name, y_true, y_pred):
    """
    Print classification metrics.
    """

    print(f"\n{name}")
    print("-" * 40)

    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_true, y_pred):.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))


def main():

    # ===================================================
    # Generate Dataset
    # ===================================================

    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )

    y = y.astype(np.float64)

    # ===================================================
    # Train / Test Split
    # ===================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # ===================================================
    # Feature Scaling
    # ===================================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ===================================================
    # Custom Logistic Regression
    # ===================================================

    print("=" * 60)
    print("Training Custom Logistic Regression")
    print("=" * 60)

    custom_model = CustomLogisticRegression()

    custom_model.fit(
        X_train,
        y_train,
        optimizer=SGD(learning_rate=0.01),
        loss_fn=BinaryCrossEntropyLoss(),
        epochs=1000,
        batch_size=32,
    )

    custom_pred = custom_model.predict(X_test)

    print("\nCustom Parameters")
    print("-----------------")
    print("Weights:")
    print(custom_model.weights)
    print("Bias:")
    print(custom_model.bias)

    evaluate_model(
        "Custom Logistic Regression (Test Set)",
        y_test,
        custom_pred,
    )

    # ===================================================
    # Scikit-Learn Logistic Regression
    # ===================================================

    print("\n" + "=" * 60)
    print("Training Scikit-Learn Logistic Regression")
    print("=" * 60)

    sklearn_model = LogisticRegression(max_iter=1000)

    sklearn_model.fit(X_train, y_train)

    sklearn_pred = sklearn_model.predict(X_test)

    print("\nScikit-Learn Parameters")
    print("-----------------------")
    print("Weights:")
    print(sklearn_model.coef_)
    print("Bias:")
    print(sklearn_model.intercept_)

    evaluate_model(
        "Scikit-Learn Logistic Regression (Test Set)",
        y_test,
        sklearn_pred,
    )


if __name__ == "__main__":
    main()