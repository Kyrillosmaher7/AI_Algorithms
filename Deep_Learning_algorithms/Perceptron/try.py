import numpy as np
from Deep_Learning_algorithms.Perceptron.perceptron import Perceptron

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([0,0,0,1])

model = Perceptron(
    epochs=20
)

model.fit(X, y)

print(model.predict(X))
