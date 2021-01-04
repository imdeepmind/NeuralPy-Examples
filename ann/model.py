from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.activation_functions import ReLU,Sigmoid
from neuralpy.loss_functions import CrossEntropyLoss
from neuralpy.optimizer import Adam

import numpy as np


# Create Model

model = Sequential()

model.add(Dense(n_nodes=1, n_inputs=3))
# model.add(ReLU())
# model.add(Dense(n_nodes=2))
# model.add(ReLU())
# model.add(Dense(n_nodes=1))
model.add(Sigmoid())

model.build()

model.compile(optimizer=Adam(), loss_function=CrossEntropyLoss(),
              metrics=["accuracy"])
print(model.summary())


# Data for XOR


x_train = np.array([[0,0,1],[0,1,1],[1,0,1]])
x_test = np.array([[1,1,1]])
y_train = np.array([0,1,1])
# y_train = np.array([[0],[1],[1]])
y_test = np.array([[0]])


print(x_train.shape, y_train.shape)
print(model.predict(x_train[2]))
model.fit(train_data=(x_train, y_train), epochs=10, batch_size=1)
