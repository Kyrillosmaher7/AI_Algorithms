import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from Deep_Learning_algorithms.SigmoidNeuron.sigmoid_neuron import SigmoidNeuron

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scale features (important for gradient descent)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train your model
model = SigmoidNeuron(
    learning_rate=0.01,
    epochs=5000
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print(f"Custom Model Accuracy: {acc:.4f}")



sk_model = LogisticRegression(max_iter=5000)
sk_model.fit(X_train, y_train)

sk_pred = sk_model.predict(X_test)

print("Scikit-Learn Accuracy:",
accuracy_score(y_test, sk_pred))