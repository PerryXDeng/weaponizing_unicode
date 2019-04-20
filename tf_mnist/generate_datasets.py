import tensorflow as tf
import numpy as np
import random


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


def compile_datasets():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x train 60k * 28 * 28
    # y train 10k * 28 * 28
    num_classes = 10
    digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
    train_pairs, train_y = create_pairs(x_train, digit_indices)
    digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
    test_pairs, test_y = create_pairs(x_test, digit_indices)
    # 108400 pairs of training data
    # 17820 pairs of testing data
    size_train = train_pairs.shape[0]
    size_test = test_pairs.shape[0]
    x_1_train = np.reshape(train_pairs[:, 0], (size_train, 28, 28, 1))
    x_2_train = np.reshape(train_pairs[:, 1], (size_train, 28, 28, 1))
    y_train = np.reshape(train_y, (size_train, 1))
    x_1_test = np.reshape(test_pairs[:, 0], (size_test, 28, 28, 1))
    x_2_test = np.reshape(test_pairs[:, 1], (size_test, 28, 28, 1))
    y_test = np.reshape(test_y, (size_test, 1))
    return x_1_train, x_2_train, y_train, x_1_test, x_2_test, y_test


def main():
    compile_datasets()


if __name__ == "__main__":
    main()

