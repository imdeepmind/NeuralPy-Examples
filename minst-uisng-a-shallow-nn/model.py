# Dependencies
from neuralpy.models import Sequential
from neuralpy.layers import Dense
from neuralpy.regulariziers import Dropout
from neuralpy.activation_functions import ReLU
from neuralpy.loss_functions import CrossEntropyLoss
from neuralpy.optimizer import Adam

import pandas as pd
import numpy as np

# Model
model = Sequential()

model.add(Dense(n_nodes=256, n_inputs=784))
model.add(ReLU())

model.add(Dropout())

model.add(Dense(n_nodes=10))

model.build()

model.compile(optimizer=Adam(), loss_function=CrossEntropyLoss(), metrics=["accuracy"])

print(model.summary())

