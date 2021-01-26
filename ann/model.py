# Importing Dependencies

from neuralpy.models import Sequential
from neuralpy.layers.linear import Dense
from neuralpy.layers.activation_functions import ReLU
from neuralpy.loss_functions import MSELoss
from neuralpy.optimizer import Adam
import numpy as np


# Creating  Model
'''
This example will create an ann(Artifical Neural Network)
for a 3 input XOR logic

'''

model = Sequential()
model.add(Dense(n_nodes=1, n_inputs=3))
model.add(ReLU())
model.add(Dense(n_nodes=2))
model.add(ReLU())
model.add(Dense(n_nodes=1))
model.add(ReLU())

# Building the Model
model.build()

# Compiling
model.compile(optimizer=Adam(), loss_function=MSELoss(), metrics=["accuracy"])
print(model.summary())

# Data for XOR

x_train = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1]])
x_test = np.array([[1, 1, 1]])
y_train = np.array([[0], [1], [1]], dtype=np.float32)
y_test = np.array([[0]], dtype=np.float32)

# Training the model
model.fit(train_data=(x_train, y_train), epochs=20, batch_size=1)

# Prediction
print(model.predict(x_test[0]))
