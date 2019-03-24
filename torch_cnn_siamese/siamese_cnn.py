import torch_cnn_siamese.data_prep as util
import torch_cnn_siamese.config as config

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import imageio
import numpy as np


# Datasets
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())

testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

# Loaders for Datasets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)



def conv_layer(in_channels, out_channels, kernel, pool=True):
    """
    Represents a single conv layer within the network, consisting of a 2D convolution, activating it with
    ReLU, and then applying a 2D maxpool
    :param in_channels: input number of filters for the layer
    :param out_channels: output number of filters for the layer
    :param kernel: size of each kernel
    :return: nn.Sequential of the layer
    """
    if pool:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
    else:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, padding=1),
            nn.ReLU()
        )
    return layer


class Net(nn.Module):
    def __init__(self):
        """
        Initialization for the network, defining the hidden layers of the model
        """
        super(Net, self).__init__()
        self.conv1 = conv_layer(config.NUM_CHANNELS, 64, 3)
        self.conv2 = conv_layer(64, 128, 3)
        self.conv3 = conv_layer(128, 256, 3)
        self.conv4 = conv_layer(256, 256, 3, False)
        self.fc = nn.Linear(2304, 2304)

    def forward(self, x):
        """
        Siamese CNN model with layers of 2D Conv + ReLU combined with a MaxPool2D
        :param x: input to the network
        :return: output of the network
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 2304)
        x = self.fc(x)
        return torch.sigmoid(x)


net = Net()
cross = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=.0001)


def train(num_epoch):
    """
    Function that handles the training loop of the network
    :param num_epoch: number of times to loop through the dataset
    :return: None
    """
    for epoch in range(num_epoch):
        for i, data in enumerate(trainloader, 0):
            image, label = data
            image1, label1 = image[:len(image) // 2], label[:len(label) // 2]
            image2, label2 = image[len(image) // 2:], label[:len(label) // 2]

            # Zero grad
            optimizer.zero_grad()

            # forward, backward, optimize
            output = net(image1)
            output_2 = net(image2)

            square_mean = torch.mul(output - output_2, 2)

            loss = cross(square_mean, label1)
            loss.backward()
            optimizer.step()

            # print results
            print("Loss at iter", i, "in epoch", epoch, ": ", loss.data)
    print("Finished training")


train(1)