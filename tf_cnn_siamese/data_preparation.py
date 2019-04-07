import tf_cnn_siamese.configurations as conf
import tensorflow as tf
import numpy as np


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


def inputs_placeholders():
  x_1 = tf.placeholder(conf.DTYPE, shape=(conf.BATCH_SIZE,
                                          conf.IMG_X, conf.IMG_Y, 1), name="x_1")
  x_2 = tf.placeholder(conf.DTYPE, shape=(conf.BATCH_SIZE,
                                          conf.IMG_X, conf.IMG_Y, 1), name="x_2")
  labels = tf.placeholder(conf.DTYPE, shape=(conf.BATCH_SIZE, 1), name="labels")
  return x_1, x_2, labels


