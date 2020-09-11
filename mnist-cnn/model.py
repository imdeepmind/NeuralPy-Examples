from neuralpy.models import Sequential
from neuralpy.layers import Conv2D, Dense, Flatten
from neuralpy.activation_functions import ReLU,Softmax
from neuralpy.loss_functions import CrossEntropyLoss
from neuralpy.optimizer import SGD
import torch
import torchvision
from torchvision import datasets, transforms

# Create a Sequential model Instance
model = Sequential()

#Build your network
model.add(Conv2D(input_shape=(1,28,28), filters=128, kernel_size=3))
model.add(ReLU())
model.add(Conv2D(filters=64, kernel_size=3))
model.add(ReLU())
model.add(Conv2D(filters=32, kernel_size=3))
model.add(ReLU())
model.add(Flatten())
model.add(Dense(n_nodes=10))

model.build()
model.compile(optimizer=SGD(), loss_function=CrossEntropyLoss(), metrics=["accuracy"])
print(model.summary())

#Get the MNIST dataset 
train_set = torchvision.datasets.MNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
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
#Train the model
model.fit(train_data=(train_imgs,train_labels), validation_data=(test_imgs, test_labels), epochs=10, batch_size=10)
