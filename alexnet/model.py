from neuralpy.models import Sequential
from neuralpy.layers.linear import Dense
from neuralpy.layers.convolutional import Conv2D
from neuralpy.layers.activation_functions import ReLU,Softmax
from neuralpy.layers.pooling import MaxPool2D
from neuralpy.layers.other import Flatten
from neuralpy.loss_functions import CrossEntropyLoss
from neuralpy.optimizer import SGD,Adam

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create a Sequential model Instance
model = Sequential()

model.add(Conv2D(input_shape=(1,224,224), filters=96, kernel_size=11, stride=4))
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
model.add(Dense(n_nodes=4096))
model.add(ReLU())
model.add(Dense(n_nodes=4096))
model.add(ReLU())
model.add(Dense(n_nodes=10))
model.add(Softmax())

model.build()
model.compile(optimizer=Adam(), loss_function=CrossEntropyLoss(), metrics=["accuracy"])
print(model.summary())

# Get the training Data
train_set = datasets.MNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
)

# Load the dataset from pytorch's Dataloader function
train_loader  = torch.utils.data.DataLoader(train_set, batch_size=1000)
#Get the data
mnist_data = next(iter(train_loader))
#Split into train and test set
train_imgs = mnist_data[0][:500]
train_labels = mnist_data[1][:500]
test_imgs = mnist_data[0][500:]
test_labels = mnist_data[1][500:]
# Train Model
model.fit(train_data=(train_imgs,train_labels), epochs=5, validation_data=(test_imgs,test_labels),batch_size=1)