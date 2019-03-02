import tf_cnn_siamese.configurations as conf
import tensorflow as tf


def single_cnn(x, conv1_weights, conv1_biases, conv2_weights, conv2_biases,
               fc1_weights, fc1_biases, dropout = False):
  # conv -> relu -> pool -> conv -> relu -> pool -> fully connected
  conv = tf.nn.conv2d(x,
                      conv1_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

  conv = tf.nn.conv2d(pool,
                      conv2_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

  # Reshape the feature map cuboid into a matrix for fc layers
  pool_shape = pool.get_shape().as_list()
  features = tf.reshape(
    pool,
    [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

  # last fc_weights determine output dimensions
  fc = tf.nn.sigmoid(tf.matmul(features, fc1_weights) + fc1_biases)

  # for actual training
  if dropout:
    fc = tf.nn.dropout(fc, conf.DROP_RATE)
  return fc
