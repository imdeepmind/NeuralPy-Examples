# Importing dependencies
import numpy as np

from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.optimizer import Adam
from neuralpy.loss_functions import MSELoss

# Random seed for numpy
np.random.seed(1969)

# Generating the data
X_train = np.random.rand(100, 1) * 10
y_train = X_train + 5 *np.random.rand(100, 1)

X_validation = np.random.rand(100, 1) * 10
y_validation = X_validation + 5 * np.random.rand(100, 1)

X_test = np.random.rand(10, 1) * 10
y_test = X_test + 5 * np.random.rand(10, 1)

# Making the model
model = Sequential()
model.add(Dense(n_nodes=1, n_inputs=1))

# Compiling the model
model.compile(optimizer=Adam(), loss_function=MSELoss())

# Printing model summary
model.summary()

# Training the model
history = model.fit(train_data=(X_train, y_train), test_data=(X_validation, y_validation), epochs=300, batch_size=4)

# Predicting some values
evaluated = model.evaluate(X=X_test, y=y_test, batch_size=4)

print(evaluated)