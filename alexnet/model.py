from neuralpy.models import Sequential
from neuralpy.layers.linear import Dense
from neuralpy.layers.convolutional import Conv2D
from neuralpy.layers.activation_functions import ReLU,Softmax
from neuralpy.layers.pooling import MaxPool2D
from neuralpy.layers.other import Flatten
from neuralpy.loss_functions import CrossEntropyLoss
from neuralpy.optimizer import SGD

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

model = Sequential()

model.add(Conv2D(input_shape=(3,224,224), filters=96, kernel_size=11, stride=4))
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

train_data = datasets.CIFAR100(root="datset/",train=True,download=True, transform=transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)

test_data = datasets.CIFAR100(root="datset/",train=False,download=True, transform=transforms.Compose([transforms.CenterCrop(224),transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_data, batch_size=2)

next_data_sample = next(iter(train_loader))
print(next_data_sample[1].shape, next_data_sample[1].shape)

exit()

train_samples_imgs = next_data_sample[0][0:500]
train_samples_lbs = next_data_sample[1][0:500]

model.fit(train_data=(train_samples_imgs,train_samples_lbs), epochs=5, validation_data=(train_samples_imgs,train_samples_lbs),batch_size=2)