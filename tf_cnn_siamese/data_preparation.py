import tf_cnn_siamese.configurations as conf
import tensorflow as tf


def inputs_placeholders():
  x_1 = tf.placeholder(conf.DTYPE, shape=(conf.BATCH_SIZE,
                                          conf.IMG_X, conf.IMG_Y))
  x_2 = tf.placeholder(conf.DTYPE, shape=(conf.BATCH_SIZE,
                                          conf.IMG_X, conf.IMG_Y))
  labels = tf.placeholder(conf.DTYPE, shape =(conf.BATCH_SIZE,))
  return x_1, x_2, labels
