import tf_cnn_siamese.configurations as conf
import tf_cnn_siamese.data_preparation as dp
import tensorflow as tf
import numpy as np


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


def construct_logits_model(x_1, x_2, conv1_weights, conv1_biases, conv2_weights,
                           conv2_biases, fc1_weights, fc1_biases,
                           fcj_weights, fcj_biases, dropout=False):
  # actual neural nets (twin portion)
  twin_1 = single_cnn(x_1, conv1_weights, conv1_biases, conv2_weights, conv2_biases
                      , fc1_weights, fc1_biases, dropout)
  twin_2 = single_cnn(x_2, conv1_weights, conv1_biases, conv2_weights, conv2_biases
                      , fc1_weights, fc1_biases, dropout)
  # logits on squared difference (joined portion)
  sq_diff = tf.squared_difference(twin_1, twin_2)
  logits = tf.matmul(sq_diff, fcj_weights) + fcj_biases
  return logits


def construct_full_model(x_1, x_2, conv1_weights, conv1_biases, conv2_weights,
                         conv2_biases, fc1_weights, fc1_biases,
                         fcj_weights, fcj_biases):
  logits = construct_logits_model(x_1, x_2, conv1_weights, conv1_biases,
                                  conv2_weights, conv2_biases, fc1_weights,
                                  fc1_biases, fcj_weights, fcj_biases)
  # actual neural network
  return tf.nn.sigmoid(logits)


def construct_loss_optimizer(x_1, x_2, labels, conv1_weights, conv1_biases,
                             conv2_weights, conv2_biases, fc1_weights, fc1_biases,
                             fcj_weights, fcj_biases, dropout=False, lagrange=False):
  logits = construct_logits_model(x_1, x_2, conv1_weights, conv1_biases,
                                  conv2_weights, conv2_biases, fc1_weights,
                                  fc1_biases, fcj_weights, fcj_biases, dropout)
  # cross entropy loss on sigmoids of joined output and labels
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                                logits=logits))
  if lagrange:
    # constraints on sigmoid layers
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fcj_weights) + tf.nn.l2_loss(fcj_biases))
    loss += conf.LAMBDA * regularizers

  # setting up the optimization
  batch = tf.Variable(0, dtype=conf.DTYPE)
  batch_total = labels.shape[0]
  learning_rate = tf.train.exponential_decay(
      conf.BASE_LEARNING_RATE,
      batch * conf.BATCH_SIZE,  # Current index into the dataset.
      batch_total,
      conf.DECAY_RATE,                # Decay rate.
      staircase=True)
  # accumulation = momentum * accumulation + gradient
  # every epoch: variable -= learning_rate * accumulation
  trainer = tf.train.MomentumOptimizer(learning_rate, conf.MOMENTUM)\
      .minimize(loss, global_step=batch)
  return trainer


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 1 - (np.sum(np.argmax(predictions, 1) == labels)
              / predictions.shape[0])


def batch_validate(set_1, set_2, labels, conv1_weights, conv1_biases,
                   conv2_weights, conv2_biases, fc1_weights, fc1_biases,
                   fcj_weights, fcj_biases):
  x_1, x_2, labels = dp.inputs_placeholders()
  model = construct_full_model(x_1, x_2, conv1_weights, conv1_biases,
                                     conv2_weights, conv2_biases, fc1_weights,
                                     fc1_biases, fcj_weights, fcj_biases)
  size = set_1.shape[0]
  if size < conf.BATCH_SIZE:
    raise ValueError("batch size for validation larger than dataset: %d" % size)
  predictions = np.ndarray(shape=(size), dtype=np.float32)
  for begin in range(0, size, conf.BATCH_SIZE):
    end = begin + conf.BATCH_SIZE
    if end <= size:
      predictions[begin:end] = tf.Session.run(model,
                                              feed_dict={
                                                  x_1: set_1[begin:end, ...],
                                                  x_2: set_2[begin:end, ...]})
    else:
      batch_predictions = tf.Session.run(model,
                                         feed_dict={
                                            x_1: set_1[-conf.BATCH_SIZE:, ...],
                                            x_2: set_2[-conf.BATCH_SIZE:, ...]})
      predictions[begin:] = batch_predictions[begin - size:, :]
  return predictions


def run_training_session(set_1, set_2, labels, dropout=False, lagrange=False):
  # inputs
  x_1, x_2, labels = dp.inputs_placeholders()

  # twin portion variables
  # 5x5 filter, depth 32.
  conv1_weights = tf.Variable(tf.truncated_normal([5, 5, 32],
                              stddev=0.1, seed=conf.SEED, dtype=conf.DTYPE))
  conv1_biases = tf.Variable(tf.zeros([32], dtype=conf.DTYPE))
  conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64],
                              stddev=0.1, seed=conf.SEED, dtype=conf.DTYPE))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=conf.DTYPE))
  # fully connected
  num_features = 256
  fc1_weights = tf.Variable(tf.truncated_normal(
                            [conf.IMG_X // 4 * conf.IMG_Y // 4 * 64, num_features],
                            stddev=0.1, seed=conf.SEED, dtype=conf.DTYPE))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=conf.DTYPE))

  # joined portion variables, turns num_features into 1 probability
  fcj_weights = tf.Variable(tf.truncated_normal([num_features, 1],
                                                stddev=0.1,
                                                seed=conf.SEED,
                                                dtype=conf.DTYPE))
  fcj_biases = tf.Variable(tf.constant(0.1, shape=[1], dtype=conf.DTYPE))
  optimizer = construct_loss_optimizer(x_1, x_2, labels, conv1_weights, conv1_biases,
                                       conv2_weights, conv2_biases, fc1_weights,
                                       fc1_biases, fcj_weights, fcj_biases, dropout,
                                       lagrange)
