
from Lib.Initializers.he_normal import HeNormal
from Lib.Initializers.xavier_uniform import XavierUniform
from Lib.Layers.Dense import Dense
from Lib.Layers.ActivationLayers.relu import ReLULayer
from Lib.Layers.ActivationLayers.sgmoid import SigmoidLayer
from Lib.Model.sequential import Sequential
from Lib.Optimizars.stochastic_gradient_descent import SGD
from Lib.Losses.binary_cross_entropy import BinaryCrossEntropyLoss


model = Sequential([
    Dense(
        2,
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


model.summary()

import numpy as np


X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])


y = np.array([
    [0],
    [1],
    [1],
    [0]
])

model.compile(
    optimizer=SGD(
        learning_rate=0.1
    ),

    loss=BinaryCrossEntropyLoss()
)

history = model.fit(
    X,
    y,
    epochs=5000
)

pred = model.predict(X)

print(pred)
print(pred > 0.5)