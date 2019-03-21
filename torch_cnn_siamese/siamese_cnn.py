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
        self.conv3 = conv_layer(128, 128, 3)
        self.conv4 = conv_layer(128, 256, 3)
        self.conv5 = conv_layer(256, 4096, 3, False)

    def forward(self, x):
        """
        Siamese CNN model with layers of 2D Conv + ReLU combined with a MaxPool2D
        :param x: input to the network
        :return: output of the network
        """
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.conv4(x)
        print(x.shape)
        x = self.conv5(x)
        print(x.shape)
        x = x.view(-1, 4096)
        print(x.shape)
        return torch.sigmoid(x)


net = Net()
cross = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=.0001)

image = torch.randn(1, 1, 28, 28)
output = net(image)

image = torch.randn(1, 1, 28, 28)
output_2 = net(image)

diff = output - output_2



loss = cross(torch.mul(diff, diff), )

loss.backward()
optimizer.step()
