import tf_cnn_siamese.configurations as conf
import tf_mnist.generate_datasets as tfdata
import tensorflow as tf
import numpy as np


def generate_normalized_data(num_pairs):
  # pairs of tensors of images with dimention specified in conf
  x_1 = np.reshape(np.random.rand(num_pairs, conf.IMG_X, conf.IMG_Y),
                   (num_pairs, conf.IMG_X, conf.IMG_Y, 1))
  x_2 =  np.reshape(np.random.rand(num_pairs, conf.IMG_X, conf.IMG_Y),
                   (num_pairs, conf.IMG_X, conf.IMG_Y, 1))
  # shifting the range from (0, 1) to (-0.5, 0.5)
  x_1 -= 0.5
  x_2 -= 0.5
  labels = np.random.choice(a=[0, 1], size=num_pairs, p=[0.48, 0.52])
  return x_1, x_2, np.reshape(labels, (num_pairs, 1))


def generate_features(num_pairs):
  # pairs of fake features
  twin_1 = np.reshape(np.random.rand(num_pairs, conf.NUM_FC_NEURONS),
                     (num_pairs, conf.NUM_FC_NEURONS))
  twin_2 = np.reshape(np.random.rand(num_pairs, conf.NUM_FC_NEURONS),
                     (num_pairs, conf.NUM_FC_NEURONS))
  return twin_1, twin_2


def training_inputs_placeholders():
  x_1 = tf.placeholder(conf.DTYPE, shape=conf.TRAIN_INPUT_SHAPE, name="x_1")
  x_2 = tf.placeholder(conf.DTYPE, shape=conf.TRAIN_INPUT_SHAPE, name="x_2")
  labels = tf.placeholder(conf.DTYPE, shape=(conf.TRAIN_BATCH_SIZE, 1), name="labels")
  return x_1, x_2, labels


def test_inputs_placeholders():
  x_1 = tf.placeholder(conf.DTYPE, shape=conf.TEST_INPUT_SHAPE, name="x_1")
  x_2 = tf.placeholder(conf.DTYPE, shape=conf.TEST_INPUT_SHAPE, name="x_2")
  labels = tf.placeholder(conf.DTYPE, shape=(conf.TEST_BATCH_SIZE, 1), name="labels")
  return x_1, x_2, labels


def test_features_placeholders():
  twin_1 = tf.placeholder(conf.DTYPE,
                          shape=(conf.TEST_BATCH_SIZE, conf.NUM_FC_NEURONS),
                          name="twin_1")
  twin_2 = tf.placeholder(conf.DTYPE,
                          shape=(conf.TEST_BATCH_SIZE, conf.NUM_FC_NEURONS),
                          name="twin_2")
  return twin_1, twin_2


def get_mnist_dataset():
  tset1, tset2, tlabels, vset1, vset2, vlabels = tfdata.compile_transformed_float32_datasets()
  if conf.DATA_FORMAT == 'NCHW':
    transform = [0, 3, 1, 2]
    tset1 = np.transpose(tset1, transform)
    tset2 = np.transpose(tset2, transform)
    vset1 = np.transpose(vset1, transform)
    vset2 = np.transpose(vset2, transform)
  return tset1, tset2, tlabels, vset1, vset2, vlabels
