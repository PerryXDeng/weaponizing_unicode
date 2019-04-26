import torch_cnn_siamese.config as conf
import torchvision
import torchvision.transforms as transforms
import torch
import random
import numpy as np


def generate_normalized_data(num_pairs):
    # pairs of tensors of images with dimention specified in conf
    x_1 = np.random.rand(num_pairs, conf.IMG_X, conf.IMG_Y)
    x_2 = np.random.rand(num_pairs, conf.IMG_X, conf.IMG_Y)
    # shfiting the range from (0, 1) to (-0.5, 0.5)
    x_1 -= 0.5
    x_2 -= 0.5
    labels = np.random.choice(a=[0, 1], size=num_pairs, p=[0.48, 0.52])
    return x_1, x_2, labels


def create_pairs(x, digit_indices):
    random.seed(0)
    num_classes = 10
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]	# pair of data of the same class
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes# random class
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]   # pair of data of two different class
            labels += [1, 0]            # two consecutive pairs
    return np.array(pairs), np.array(labels)


def generate_dataset():
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    # Loaders for Datasets
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64)

    print(trainloader[0])


generate_dataset()