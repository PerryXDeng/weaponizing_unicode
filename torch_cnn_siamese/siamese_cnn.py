import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch.onnx as onnx
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tf_cnn_siamese.data_preparation import get_mnist_dataset
from torch_cnn_siamese.config import BATCH_SIZE, EPOCHS_PER_VALIDATION, DECAY_RATE


# MNIST Datasets
trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

# Loaders for Datasets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64)

# MNIST One Shot Dataset
# tset1, tset2, tlabels, vset1, vset2, vlabels = get_mnist_dataset()


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
        self.conv1 = conv_layer(1, 64, 3)
        self.conv2 = conv_layer(64, 128, 3)
        self.conv3 = conv_layer(128, 256, 3)
        self.conv4 = conv_layer(256, 256, 3, False)
        self.fc = nn.Linear(2304, 2304)
        self.final = nn.Linear(2304, 1)

    def forward(self, params):
        """
        Siamese CNN model with layers of 2D Conv + ReLU combined with a MaxPool2D
        :param params: input tuple to the network, 'genuine' and 'fake' images
        :return: output of the network
        """
        x, y = params[0], params[1]

        # Putting images through SiameseCNN
        x, y = self.conv1(x), self.conv1(y)
        x, y = self.conv2(x), self.conv2(y)
        x, y = self.conv3(x), self.conv3(y)
        x, y = self.conv4(x), self.conv4(y)

        # Reshaping and getting FC layer output
        x, y = x.view(-1, 2304), y.view(-1, 2304)
        xfin = torch.sigmoid(self.fc(x))
        yfin = torch.sigmoid(self.fc(y))

        # Performing square mean and squaring
        square_mean = torch.pow(xfin.sub(yfin), 2)
        result = self.final(square_mean)
        return torch.sigmoid(result)


net = Net()
cross = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=.001, weight_decay=DECAY_RATE)


def train(num_epoch):
    """
    Function that handles the training loop of the network
    :param num_epoch: number of times to loop through the dataset
    :return: None
    """
    data_size = testloader.shape[0]
    total = int(num_epoch * data_size)
    num_steps = total // BATCH_SIZE
    steps_per_epoch = data_size / BATCH_SIZE
    validation_interval = int(EPOCHS_PER_VALIDATION * steps_per_epoch)
    print("Training Started")

    start_time = time.time()
    for epoch in range(num_epoch):
        for step in range(num_steps):
            # offset of the current minibatch
            offset = (step * BATCH_SIZE) % (data_size - BATCH_SIZE)
            batch_x1 = tset1[offset:(offset + BATCH_SIZE), ...]
            batch_x2 = tset2[offset:(offset + BATCH_SIZE), ...]
            batch_labels = tlabels[offset:(offset + BATCH_SIZE)]

            # Zero grad
            optimizer.zero_grad()

            # forward, backward, optimize
            output = net((batch_x1, batch_x2))
            loss = cross(output, batch_labels)
            loss.backward()
            optimizer.step()

            # print results
            if step % validation_interval == 0:
                for param_group in optimizer.param_groups:
                    lr = param_group['lr']
                print('Step %d (epoch %.2f), %.4f s' % (step, epoch, time.time() - start_time))
                print('Minibatch Loss: %.3f, learning rate: %.10f' % (loss, lr))
                print('Training Accuracy: %.3f' % train_net_accuracy())
                print('Validation Accuracy: %.3f' % validate_set_accuracy())

    print("Finished training in %.4f s" % (time.time() - start_time))


def train_net_accuracy():
    """
    Function to handle testing the network for accuracy on the training set
    :return: None
    """
    total = tset1.shape[0]
    steps = total // BATCH_SIZE

    correct = 0
    with torch.no_grad():
        for step in range(steps):
            # offset of the current minibatch
            offset = (step * BATCH_SIZE) % (total - BATCH_SIZE)
            batch_x1 = tset1[offset:(offset + BATCH_SIZE), ...]
            batch_x2 = tset2[offset:(offset + BATCH_SIZE), ...]
            batch_labels = tlabels[offset:(offset + BATCH_SIZE)]

            output = net((batch_x1, batch_x2))
            print(output)
            correct += np.sum(output == batch_labels)

    return 100 * correct / total


def validate_set_accuracy():
    """
        Function to handle testing the network for accuracy on the test set
        The validate images are ones the network hasn't seen
        :return: None
        """
    total = vset1.shape[0]
    steps = total // BATCH_SIZE

    correct = 0
    with torch.no_grad():
        for step in range(steps):
            # offset of the current minibatch
            offset = (step * BATCH_SIZE) % (total - BATCH_SIZE)
            batch_x1 = vset1[offset:(offset + BATCH_SIZE), ...]
            batch_x2 = vset2[offset:(offset + BATCH_SIZE), ...]
            batch_labels = vlabels[offset:(offset + BATCH_SIZE)]

            output = net((batch_x1, batch_x2))
            print(output)
            correct += np.sum(output == batch_labels)

    return 100 * correct / total


train_net_accuracy()
validate_set_accuracy()

train(5)

train_net_accuracy()
validate_set_accuracy()

# Model Saving for export
# torch.onnx.export(net, ((torch.randn((1, 1, 28, 28)), torch.randn((1, 1, 28, 28))), ), "siamese_cnn.onnx")
# torch.save(net, "siamese_cnn.pth")
