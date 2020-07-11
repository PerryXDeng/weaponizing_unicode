import tensorflow as tf
import torch
import numpy as np
import legacy_code.torch_cnn_siamese.config as conf
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
    x_1_train = np.reshape(train_pairs[:, 0], (size_train, 1, 28, 28))
    x_2_train = np.reshape(train_pairs[:, 1], (size_train, 1, 28, 28))
    y_train = np.reshape(train_y, (size_train, 1))
    x_1_test = np.reshape(test_pairs[:, 0], (size_test, 1, 28, 28))
    x_2_test = np.reshape(test_pairs[:, 1], (size_test, 1, 28, 28))
    y_test = np.reshape(test_y, (size_test, 1))
    return x_1_train, x_2_train, y_train, x_1_test, x_2_test, y_test


def compile_transformed_float32_datasets():
    x_1_train, x_2_train, y_train, x_1_test, x_2_test, y_test = compile_datasets()
    x_1_train = x_1_train.astype(np.float32)
    x_2_train = x_2_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_1_test = x_1_test.astype(np.float32)
    x_2_test = x_2_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    x_1_train = (x_1_train - 255 / 2) / 255
    x_2_train = (x_2_train - 255 / 2) / 255
    x_1_test = (x_1_test - 255 / 2) / 255
    x_2_test = (x_2_test - 255 / 2) / 255
    return x_1_train, x_2_train, y_train, x_1_test, x_2_test, y_test


def generate_normalized_data(num_pairs):
  # pairs of tensors of images with dimention specified in conf
  x_1 = np.reshape(np.random.rand(num_pairs, conf.IMG_X, conf.IMG_Y),
                   (num_pairs, conf.IMG_X, conf.IMG_Y, 1))
  x_2 =  np.reshape(np.random.rand(num_pairs, conf.IMG_X, conf.IMG_Y),
                   (num_pairs, conf.IMG_X, conf.IMG_Y, 1))
  # shfiting the range from (0, 1) to (-0.5, 0.5)
  x_1 -= 0.5
  x_2 -= 0.5
  labels = np.random.choice(a=[0, 1], size=num_pairs, p=[0.48, 0.52])
  return x_1, x_2, np.reshape(labels, (num_pairs, 1))


def get_mnist_dataset():
  tset1, tset2, tlabels, vset1, vset2, vlabels = compile_transformed_float32_datasets()
  if conf.DATA_FORMAT == 'NCHW':
    transform = [0, 3, 1, 2]
    tset1 = np.transpose(tset1, transform)
    tset2 = np.transpose(tset2, transform)
    vset1 = np.transpose(vset1, transform)
    vset2 = np.transpose(vset2, transform)


  # Converting to PyTorch Tensors
  if torch.cuda.is_available():
      torch.set_default_tensor_type('torch.cuda.FloatTensor')
  else:
      torch.set_default_tensor_type('torch.FloatTensor')

  tset1 = torch.from_numpy(tset1)
  tset2 = torch.from_numpy(tset2)
  tlabels = torch.from_numpy(tlabels)
  vset1 = torch.from_numpy(vset1)
  vset2 = torch.from_numpy(vset2)
  vlabels = torch.from_numpy(vlabels)

  return tset1, tset2, tlabels, vset1, vset2, vlabels
