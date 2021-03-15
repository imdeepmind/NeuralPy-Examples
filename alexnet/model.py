from neuralpy.models import Sequential
from neuralpy.layers.linear import Dense
from neuralpy.layers.convolutional import Conv2D
from neuralpy.layers.activation_functions import ReLU,Softmax
from neuralpy.layers.pooling import MaxPool2D
from neuralpy.layers.other import Flatten
from neuralpy.loss_functions import CrossEntropyLoss
from neuralpy.optimizer import SGD

model = Sequential()

model.add(Conv2D(input_shape=(1,28,28), filters=96, kernel_size=11, stride=4))
model.add(ReLU())
model.add(MaxPool2D(kernel_size=3, stride=2))
model.add(ReLU())
model.add(Conv2D(filters=256, kernel_size=5, stride=1, padding=2))
model.add(ReLU())
model.add(MaxPool2D(kernel_size=3, stride=2))
model.add(ReLU())
model.add(Conv2D(filters=384, kernel_size=3, stride=1, padding=1))
model.add(ReLU())
model.add(Conv2D(filters=384, kernel_size=3, stride=1, padding=1))
model.add(ReLU())
model.add(Conv2D(filters=256, kernel_size=3, stride=1, padding=1))
model.add(ReLU())
model.add(MaxPool2D(kernel_size=3, stride=2))
model.add(ReLU())
model.add(Flatten())
# model.add(Dense(n_nodes=9216))
# model.add(ReLU())
model.add(Dense(n_nodes=4096))
model.add(ReLU())
model.add(Dense(n_nodes=4096))
model.add(ReLU())
model.add(Dense(n_nodes=10))
# model.add(Softmax()))

model.build()
model.compile(optimizer=SGD(), loss_function=CrossEntropyLoss(), metrics=["accuracy"])
print(model.summary())
