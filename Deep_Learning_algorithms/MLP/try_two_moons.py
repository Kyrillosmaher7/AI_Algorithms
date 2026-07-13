import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from Lib.Initializers.he_normal import HeNormal
from Lib.Initializers.xavier_uniform import XavierUniform
from Lib.Layers.Dense import Dense
from Lib.Layers.ActivationLayers.relu import ReLULayer
from Lib.Layers.ActivationLayers.sgmoid import SigmoidLayer
from Lib.Model.sequential import Sequential
from Lib.Optimizars.stochastic_gradient_descent import SGD
from Lib.Losses.binary_cross_entropy import BinaryCrossEntropyLoss




# -----------------------------
# Create Dataset
# -----------------------------

X, y = make_moons(
    n_samples=1000,
    noise=0.1,
    random_state=42
)

# Convert y shape
y = y.reshape(-1, 1)


# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -----------------------------
# Build MLP
# -----------------------------
model = Sequential([

    Dense(
        2,
        16,
        initializer=HeNormal()
    ),

    ReLULayer(),

    Dense(
        16,
        8,
        initializer=HeNormal()
    ),

    ReLULayer(),

    Dense(
        8,
        1,
        initializer=XavierUniform()
    ),

    SigmoidLayer()
])


model.compile(
    optimizer=SGD(
        learning_rate=0.01
    ),

    loss=BinaryCrossEntropyLoss()
)


model.fit(
    X_train,
    y_train,
    epochs=3000
)


# -----------------------------
# Predict
# -----------------------------

probabilities = model.predict(X_test)

predictions = (probabilities >= 0.5).astype(int)


# -----------------------------
# Evaluation
# -----------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)


print("=" * 50)
print("Two Moons Classification")
print("=" * 50)

print("Accuracy:", accuracy)


print("\nSamples:")

for i in range(10):
    print(
        f"True: {y_test[i][0]} "
        f"| Probability: {probabilities[i][0]:.4f} "
        f"| Prediction: {predictions[i][0]}"
    )
print("Model Summary\n")
print(model.summary())