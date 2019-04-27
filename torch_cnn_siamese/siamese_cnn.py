import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from torch_cnn_siamese.generate_datasets import get_mnist_dataset
from torch_cnn_siamese.config import BATCH_SIZE, EPOCHS_PER_VALIDATION, THRESHOLD, LAMBDA, DECAY_RATE


# Argument parsing for CMD
parser = argparse.ArgumentParser(description="Hyper-parameters for network.")
parser.add_argument('-b', '--batch', action='store', type=int, default=BATCH_SIZE)
parser.add_argument('-g', '--gpu', action='store', type=bool, default=True)
parser.add_argument('-e', '--epochs', action='store', type=int, default=5)
parser.add_argument('-p', '--path', action='store', type=str, default=str(time.time()))
parser.add_argument('-l2', '--lagrange', action='store', type=bool, default=True)
args = parser.parse_args()

print(args)

# MNIST One Shot dataset
tset1, tset2, tlabels, vset1, vset2, vlabels = get_mnist_dataset()

# GPU setting
if torch.cuda.is_available() and args.gpu:
    print("GPU: True")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(0)
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def conv_layer(in_channels, out_channels, kernel, pool=True):
    """
    Represents a single conv layer within the network, consisting of a 2D convolution, activating it with
    ReLU, and then applying a 2D maxpool
    :param in_channels: input number of filters for the layer
    :param out_channels: output number of filters for the layer
    :param kernel: size of each kernel
    :param pool: whether maxpool layer is included
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
        Computes an element-wise square mean between the network outputs before putting them into a
        FC layer
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


def l2_loss(m):
    """
    L2 Regularization function, computes half the L2 norm of a tensor w/o the sqrt
    :param m: vector to compute
    :return: half the L2 norm
    """
    loss = torch.sum(torch.pow(m, 2))
    return loss / 2


net = Net().cuda()

cross = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, DECAY_RATE)


def train(num_epoch):
    """
    Function that handles the training loop of the network
    :param num_epoch: number of times to loop through the dataset
    :return: None
    """
    data_size = tset1.shape[0]
    total = int(num_epoch * data_size)
    num_steps = total // args.batch
    steps_per_epoch = data_size / args.batch
    validation_interval = int(EPOCHS_PER_VALIDATION * steps_per_epoch)
    print("Training Started...")

    start_time = time.time()
    for step in range(num_steps):
        # offset of the current minibatch
        offset = (step * args.batch) % (data_size - args.batch)
        batch_x1 = tset1[offset:(offset + args.batch), ...]
        batch_x2 = tset2[offset:(offset + args.batch), ...]
        batch_labels = tlabels[offset:(offset + args.batch)]

        # Zero grad
        optimizer.zero_grad()

        # convert to gpu
        if args.gpu:
            batch_x1 = batch_x1.to(0)
            batch_x2 = batch_x2.to(0)
            batch_labels = batch_labels.to(0)

        # forward, backward, optimize
        output = net((batch_x1, batch_x2))
        loss = cross(output, batch_labels)

        if args.lagrange:
            # constraints on sigmoid layers
            regularizers = l2_loss(net.fc.weight) + l2_loss(net.fc.bias) + \
                           l2_loss(net.final.weight) + l2_loss(net.final.bias)
            loss += LAMBDA * regularizers

        loss.backward()
        optimizer.step()

        # print results
        if step % validation_interval == 0:
            epoch = float(step) / steps_per_epoch
            scheduler.step(epoch)    # Decay LR at new epoch

            for param_group in optimizer.param_groups:
                lr = param_group['lr']

            t_accuracy, t_precision, t_recall, t_f1 = train_net_accuracy(tset1, tset2, tlabels)
            v_accuracy, v_precision, v_recall, v_f1 = train_net_accuracy(vset1, vset2, vlabels)

            print('Step %d (epoch %.2f), %.4f s' % (step, epoch, time.time() - start_time))
            print('Minibatch Loss: %.3f, Learning Rate: %.5f' % (float(loss), lr))
            print('Training Accuracy: %.3f' % t_accuracy)
            print('Training Recall: %.1f' % t_recall)
            print('Training Precision: %.1f' % t_precision)
            print('Training F1: %.1f' % t_f1)
            print('Validation Accuracy: %.3f' % v_accuracy)
            print('Validation Recall: %.3f' % v_recall)
            print('Validation Precision: %.3f' % v_precision)
            print('Validation F1: %.1f' % v_f1)
            print("")

    print("Finished training in %.4f s" % (time.time() - start_time))

    # Saving model
    # torch.onnx.export(net, ((torch.randn((1, 1, 28, 28)), torch.randn((1, 1, 28, 28))),), "siamese_cnn.onnx")
    torch.save(net, "checkpoints/siamese_cnn_{}.pth".format(args.path))


def train_net_accuracy(set1, set2, labels):
    """
    Function to handle testing the network for accuracy on the training set
    :return: None
    """
    total = set1.shape[0]
    steps = total // args.batch

    stats = np.zeros(6)
    with torch.no_grad():
        for step in range(steps):
            # offset of the current minibatch
            offset = (step * args.batch) % (total - args.batch)
            batch_x1 = set1[offset:(offset + args.batch), ...]
            batch_x2 = set2[offset:(offset + args.batch), ...]
            batch_labels = labels[offset:(offset + args.batch)]

            # convert to gpu
            if args.gpu:
                batch_x1 = batch_x1.to(0)
                batch_x2 = batch_x2.to(0)
                batch_labels = batch_labels.to(0)

            output = net((batch_x1, batch_x2))
            stats += calc_stats(output.cpu().detach().numpy(),
                                batch_labels.cpu().detach().numpy())

    accuracy = stats[1] / stats[0]
    precision = stats[2] / (stats[2] + stats[4])  # true positives / total positives
    recall = stats[2] / (stats[2] + stats[5])  # true positives / actual positives
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1


def calc_stats(output, labels):
    """
    Calculates statistics on the current network, including num correct and a confusion matrix
    :param output: output from the network
    :param labels: labels for true class
    :return: np array of statistics
    """
    total = output.shape[0]
    positive_predictions_indices = output > THRESHOLD
    negative_predictions_indices = output < THRESHOLD
    positive_labels_indices = np.where(labels == 1)[0]
    negative_labels_indices = np.where(labels == 0)[0]

    output[positive_predictions_indices] = 1
    output[negative_predictions_indices] = 0

    num_correct = np.sum(output == labels)
    num_false_positives = np.count_nonzero(output[negative_labels_indices])
    num_false_negatives = np.count_nonzero(output[positive_labels_indices] == 0)
    num_true_positives = positive_predictions_indices.shape[0] - num_false_positives
    num_true_negatives = negative_predictions_indices.shape[0] - num_false_negatives

    return np.asarray((total, num_correct, num_true_positives, num_true_negatives,
                       num_false_positives, num_false_negatives))


if __name__ == '__main__':
    train(args.epochs)
