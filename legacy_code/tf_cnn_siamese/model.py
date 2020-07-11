import legacy_code.tf_cnn_siamese.configurations as conf
import tensorflow as tf
import numpy as np


def construct_cnn(x, conv_weights, conv_biases, fc_weights, fc_biases,
                  dropout = False):
  """
  constructs the convolution graph for one image
  :param x: input node
  :param conv_weights: convolution weights
  :param conv_biases: relu biases for each convolution
  :param fc_weights: fully connected weights, only one set should be used here
  :param fc_biases: fully connected biases, only one set should be used here
  :param dropout: whether to add a dropout layer for the fully connected layer
  :return: output node
  """
  k = conf.NUM_POOL
  for i in range(conf.NUM_CONVS):
    x = tf.nn.conv2d(x, conv_weights[i], strides=[1, 1, 1, 1], padding='SAME',
                     data_format=conf.DATA_FORMAT)
    x = tf.nn.relu(tf.nn.bias_add(x, conv_biases[i],
                                  data_format=conf.DATA_FORMAT))
    if k > 0:
      x = tf.nn.max_pool(x, ksize=conf.POOL_KDIM,strides=conf.POOL_KDIM,
                        padding='VALID', data_format=conf.DATA_FORMAT)
    k -= 1
  # Reshape the feature map cuboids into vectors for fc layers
  features_shape = x.get_shape().as_list()
  n = features_shape[0]
  m = features_shape[1] * features_shape[2] * features_shape[3]
  features = tf.reshape(x, [n, m])
  # last fc_weights determine output dimensions
  fc = tf.nn.sigmoid(tf.matmul(features, fc_weights[0]) + fc_biases[0])
  # for actual training
  if dropout:
    fc = tf.nn.dropout(fc, conf.DROP_RATE)
  return fc


def construct_logits_model(x_1, x_2, conv_weights, conv_biases, fc_weights,
                           fc_biases, dropout=False):
  """
  constructs the logit node before the final sigmoid activation
  :param x_1: input image node 1
  :param x_2: input image node 2
  :param conv_weights: nodes for convolution weights
  :param conv_biases: nodes for convolution relu biases
  :param fc_weights: nodes for fully connected weights
  :param fc_biases: nodes for fully connected biases
  :param dropout: whether to include dropout layers
  :return: logit node
  """
  with tf.name_scope("twin_1"):
    twin_1 = construct_cnn(x_1, conv_weights, conv_biases,
                           fc_weights, fc_biases, dropout)
  with tf.name_scope("twin_2"):
    twin_2 = construct_cnn(x_2, conv_weights, conv_biases,
                           fc_weights, fc_biases, dropout)
  # logits on squared difference
  sq_diff = tf.squared_difference(twin_1, twin_2)
  logits = tf.matmul(sq_diff, fc_weights[1]) + fc_biases[1]
  return logits


def construct_full_model(x_1, x_2, conv_weights, conv_biases,fc_weights,
                         fc_biases):
  """
  constructs the graph for the neural network without loss node or optimizer
  :param x_1: input image node 1
  :param x_2: input image node 2
  :param conv_weights: nodes for convolution weights
  :param conv_biases: nodes for convolution relu biases
  :param fc_weights: nodes for fully connected weights
  :param fc_biases: nodes for fully connected biases
  :return: sigmoid output node
  """
  logits = construct_logits_model(x_1, x_2, conv_weights, conv_biases,
                                  fc_weights, fc_biases, dropout=False)
  return tf.nn.sigmoid(logits)


def construct_loss_optimizer(x_1, x_2, labels, conv_weights, conv_biases,
                             fc_weights, fc_biases, dropout=False,
                             lagrange=False):
  """
  constructs the neural network graph with the loss and optimizer node
  :param x_1: input image node 1
  :param x_2: input image node 2
  :param labels: expected output
  :param conv_weights: nodes for convolution weights
  :param conv_biases: nodes for convolution relu biases
  :param fc_weights: nodes for fully connected weights
  :param fc_biases: nodes for fully connected biases
  :param dropout: whether to use dropout
  :param lagrange: whether to apply constraints
  :return: the node for the optimizer as well as the loss
  """
  logits = construct_logits_model(x_1, x_2, conv_weights, conv_biases,
                                  fc_weights, fc_biases, dropout)
  # cross entropy loss on sigmoids of joined output and labels
  loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                     logits=logits)
  loss = tf.reduce_mean(loss_vec)
  if lagrange:
    # constraints on sigmoid layers
    regularizers = (tf.nn.l2_loss(fc_weights[0]) + tf.nn.l2_loss(fc_biases[0]) +
                    tf.nn.l2_loss(fc_weights[1]) + tf.nn.l2_loss(fc_biases[1]))
    loss += conf.LAMBDA * regularizers
  # setting up the optimization
  batch = tf.Variable(0, dtype=conf.DTYPE)

  # vanilla momentum optimizer
  # accumulation = momentum * accumulation + gradient
  # every epoch: variable -= learning_rate * accumulation
  # batch_total = labels.shape[0]
  # learning_rate = tf.train.exponential_decay(
  #     conf.BASE_LEARNING_RATE,
  #     batch * conf.BATCH_SIZE,  # Current index into the dataset.
  #     batch_total,
  #     conf.DECAY_RATE,                # Decay rate.
  #     staircase=True)
  # trainer = tf.train.MomentumOptimizer(learning_rate, conf.MOMENTUM)\
  #    .minimize(loss, global_step=batch)

  # adaptive momentum estimation optimizer
  # default params: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08
  trainer = tf.train.AdamOptimizer().minimize(loss, global_step=batch)
  return trainer, loss


def construct_joined_model(twin_1, twin_2, fc_weights, fc_biases):
  """
  constructs joined model for two sets of extracted features
  :param twin_1: features node extracted from first image
  :param twin_2: features node extracted from second image
  :param fc_weights: nodes for fully connected weights
  :param fc_biases: nodes for fully connected biases
  :return: logit node
  """
  # logits on squared difference
  sq_diff = tf.squared_difference(twin_1, twin_2)
  logits = tf.matmul(sq_diff, fc_weights[1]) + fc_biases[1]
  return tf.nn.sigmoid(logits)


def initialize_weights():
  """
  initializes the variable tensors to be trained in the neural network, decides
  network dimensions
  :return: nodes for the variables
  """
  # twin network convolution and pooling variables
  conv_weights = []
  conv_biases = []
  fc_weights = []
  fc_biases = []
  for i in range(conf.NUM_CONVS):
    if i == 0:
      inp = conf.NUM_CHANNELS
    else:
      inp = conf.NUM_FILTERS[i - 1]
    out = conf.NUM_FILTERS[i]
    conv_dim = [conf.FILTER_LEN, conf.FILTER_LEN, inp, out]
    weight_name = "twin_conv" + str(i + 1) + "_weights"
    bias_name = "twin_conv" + str(i + 1) + "_biases"
    conv_weights.append(tf.Variable(tf.truncated_normal(conv_dim, stddev=0.1,
                                    seed=conf.SEED, dtype=conf.DTYPE),
                                    name=weight_name))
    conv_biases.append(tf.Variable(tf.zeros([out], dtype=conf.DTYPE),
                                   name=bias_name))
  # twin network fullly connected variables
  inp = conf.FEATURE_MAP_SIZE
  out = conf.NUM_FC_NEURONS
  fc_weights.append(tf.Variable(tf.truncated_normal([inp, out], stddev=0.1,
                                seed=conf.SEED, dtype=conf.DTYPE),
                                name="twin_fc_weights"))
  fc_biases.append(tf.Variable(tf.constant(0.1, shape=[out], dtype=conf.DTYPE),
                               name="twin_fc_biases"))
  # joined network fully connected variables
  inp = conf.NUM_FC_NEURONS
  out = 1
  fc_weights.append(tf.Variable(tf.truncated_normal([inp, out], stddev=0.1,
                                seed=conf.SEED, dtype=conf.DTYPE),
                                name="joined_fc_weights"))
  fc_biases.append(tf.Variable(tf.constant(0.1, shape=[out], dtype=conf.DTYPE),
                               name="joined_fc_biases"))
  return conv_weights, conv_biases, fc_weights, fc_biases


def num_params():
  """
  calculates the number of parameters in the model
  :return: m, number of parameters
  """
  m = 0
  for i in range(conf.NUM_CONVS):
    if i == 0:
      inp = conf.NUM_CHANNELS
    else:
      inp = conf.NUM_FILTERS[i - 1]
    out = conf.NUM_FILTERS[i]
    conv_dim = [conf.FILTER_LEN, conf.FILTER_LEN, inp, out]
    m += np.prod(conv_dim) + np.prod(out)
  inp = conf.FEATURE_MAP_SIZE
  out = conf.NUM_FC_NEURONS
  m += inp * out + out
  inp = conf.NUM_FC_NEURONS
  out = 1
  m += inp * out + out
  return m


if __name__ == "__main__":
  print("Number of Parameters: " + str(num_params()))
