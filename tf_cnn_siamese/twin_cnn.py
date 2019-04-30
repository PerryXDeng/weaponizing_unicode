import tf_cnn_siamese.configurations as conf
import tf_cnn_siamese.data_preparation as dp
import tensorflow as tf
import numpy as np
import datetime
import time
import sys
import os


def single_cnn(x, conv_weights, conv_biases, fc_weights, fc_biases,
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
    twin_1 = single_cnn(x_1, conv_weights, conv_biases,
                        fc_weights, fc_biases, dropout)
  with tf.name_scope("twin_2"):
    twin_2 = single_cnn(x_2, conv_weights, conv_biases,
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


def calc_stats(output, labels):
  """
  calculates the confusion matrix stats for a batch
  :param output: output from session
  :param labels: expected output
  :return: np array of stats, see variable names on return statement
  """
  total = output.shape[0]
  positive_predictions_indices = output > conf.THRESHOLD
  negative_predictions_indices = output < conf.THRESHOLD
  positive_labels_indices = np.where(labels == 1)[0]
  negative_labels_indices = np.where(labels == 0)[0]
  output[positive_predictions_indices] = 1
  output[negative_predictions_indices] = 0

  num_correct = np.sum(output == labels)
  num_false_positives = np.count_nonzero(output[negative_labels_indices])
  num_false_negatives = np.count_nonzero(output[positive_labels_indices] == 0)
  num_true_positives = np.count_nonzero(positive_predictions_indices)\
                       - num_false_positives
  num_true_negatives = np.count_nonzero(negative_predictions_indices)\
                       - num_false_negatives
  return np.asarray((total, num_correct, num_true_positives, num_true_negatives,
                    num_false_positives, num_false_negatives))


def batch_validate(x1, x2, labels, conv_weights, conv_biases, fc_weights,
                   fc_biases, session):
  """
  calculates the relevant metrics for the whole input dataset by batch
  :param x1: input image node 1
  :param x2: input image node 2
  :param labels: expected output
  :param conv_weights: nodes for convolution weights
  :param conv_biases: nodes for convolution relu biases
  :param fc_weights: nodes for fully connected weights
  :param fc_biases: nodes for fully connected biases
  :param session: the tensorflow session in which the compute takes place
  :return: the relevant metrics
  """
  feed_1, feed_2, _ = dp.inputs_placeholders()
  model = construct_full_model(feed_1, feed_2, conv_weights, conv_biases,
                               fc_weights, fc_biases)
  size = x1.shape[0]
  if size < conf.BATCH_SIZE:
    raise ValueError("batch size for validation larger than dataset: %d" % size)
  stats = np.zeros(6)
  for begin in range(0, size, conf.BATCH_SIZE):
    end = begin + conf.VALIDATION_BATCH_SIZE
    if end <= size:
      stats += calc_stats(session.run(model,
                                      feed_dict={feed_1: x1[begin:end, ...],
                                                 feed_2: x2[begin:end, ...]}),
                          labels[begin:end])
    else:
      full_batch_offset = -1 * conf.VALIDATION_BATCH_SIZE
      batch_predictions = session.run(model,
                                      feed_dict={
                                      feed_1: x1[full_batch_offset:, ...],
                                      feed_2: x2[full_batch_offset:, ...]})
      stats += calc_stats(batch_predictions[begin - size:, :], labels[begin:])
  accuracy = stats[1] / stats[0]
  # use precision and recall for the "positive" that is underrepresented
  # and important to detect
  # precision, proportion of correctly detected positive values
  # the fewer actual negatives misclassified, the closer to 1
  # affected by skewed data
  precision = stats[2] / (stats[2] + stats[4]) # true positives/total positives
  # coverage of actual positives, true positive rate
  # the fewer actual positives misclassified, the closer to 1
  # affected by skewed data
  recall = stats[2] / (stats[2] + stats[5]) # true positives/actual positives
  f1 = 2 * (precision*recall) / (precision + recall)
  return accuracy, precision, recall, f1


def run_training_session(tset1, tset2, ty, vset1, vset2, vy, epochs,
                         conv_weights, conv_biases, fc_weights, fc_biases,
                         dropout=False, lagrange=False):
  """
  runs a training tensorflow session given datasets
  :param tset1: training dataset input one
  :param tset2: training dataset input two
  :param ty: training dataset labels
  :param vset1: validation set input one
  :param vset2: validation set input two
  :param vy: validation set labels
  :param epochs: number of iteration over the training dataset
  :param conv_weights: nodes for convolution weights
  :param conv_biases: nodes for convolution relu biases
  :param fc_weights: nodes for fully connected weights
  :param fc_biases: nodes for fully connected biases
  :param dropout: whether to add dropout layers
  :param lagrange: whether to add constraints
  :return: None
  """
  # input nodes
  x_1, x_2, labels = dp.inputs_placeholders()
  optimizer, l = construct_loss_optimizer(x_1, x_2, labels, conv_weights,
                                          conv_biases, fc_weights, fc_biases,
                                          dropout, lagrange)
  # creates session
  saver = tf.train.Saver()
  with tf.Session() as sess: # automatic tear down of controlled execution
    print("\n")
    tf.global_variables_initializer().run()
    print("Data Format " + conf.DATA_FORMAT)
    cuda_enabled = ('NCHW' == conf.DATA_FORMAT)
    print("CUDA Enabled: " + str(cuda_enabled))
    if input("resume from previous session (y/n):") == "y":
      path = input("enter path:")
      saver.restore(sess, path)
    logger = tf.summary.FileWriter(conf.log_dir)
    print("Training Started")

    # iterates through the training data in batch
    data_size = tset1.shape[0]
    total = int(epochs * data_size)
    num_steps = total // conf.BATCH_SIZE
    steps_per_epoch = data_size / conf.BATCH_SIZE
    validation_interval = int(conf.EPOCHS_PER_VALIDATION * steps_per_epoch)
    start_time = time.time()
    for step in range(num_steps):
      # offset of the current minibatch
      offset = (step * conf.BATCH_SIZE) % (data_size - conf.BATCH_SIZE)
      batch_x1 = tset1[offset:(offset + conf.BATCH_SIZE), ...]
      batch_x2 = tset2[offset:(offset + conf.BATCH_SIZE), ...]
      batch_labels = ty[offset:(offset + conf.BATCH_SIZE)]
      # maps batched input to graph data nodes
      feed_dict = {x_1: batch_x1, x_2: batch_x2, labels: batch_labels}
      # runs the optimizer every iteration
      sess.run(optimizer, feed_dict=feed_dict)
      # prints loss and validation error when evalidating intermittently
      if step % validation_interval == 0:
        elapsed_time = time.time() - start_time
        current_epoch = float(step) / steps_per_epoch
        loss = sess.run(l, feed_dict=feed_dict)
        t_accuracy, t_precision, t_recall, t_f1 = batch_validate(tset1, tset2,
                                                 ty, conv_weights, conv_biases,
                                                 fc_weights, fc_biases, sess)
        v_accuracy, v_precision, v_recall, v_f1 = batch_validate(vset1, vset2,
                                                 vy, conv_weights, conv_biases,
                                                 fc_weights, fc_biases, sess)
        print('\nStep %d (epoch %.2f), %.4f s'
              % (step, current_epoch, elapsed_time))
        print('Minibatch Loss: %.3f' % (float(loss)))
        print('Training Accuracy: %.3f' % t_accuracy)
        print('Training Recall: %.3f' % t_recall)
        print('Training Precision: %.3f' % t_precision)
        print('Training F1: %.3f' % t_f1)
        print('Validation Accuracy: %.3f' % v_accuracy)
        print('Validation Recall: %.3f' % v_recall)
        print('Validation Precision: %.3f' % v_precision)
        print('Validation F1: %.3f' % v_f1)
        sys.stdout.flush()
    logger.add_graph(sess.graph)
    print('\nTraining Finished')
    if input("Save Variables? (y/n): ") == "y":
      filename = datetime.datetime.now().strftime("GMT%m%d%H%M.ckpt")
      try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        path = saver.save(sess, current_dir + "/saved/" + filename)
        print("Model saved in " + path)
      except:
        print("Error saving to that path")
        path = input("Enter New Path: ")
        path = tf.train.Saver().save(sess, path)
        print("Model saved in " + path)


def initialize_weights():
  """
  initialize the variable to be trained in the neural network, decides
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


def random_training_test():
  """
  tests the graph on a random dataset
  :return: None
  """
  num_pairs = 1000
  tset1, tset2, tlabels = dp.generate_normalized_data(int(0.6 * num_pairs))
  vset1, vset2, vlabels = dp.generate_normalized_data(int(0.2 * num_pairs))
  conv_weights, conv_biases, fc_weights, fc_biases = initialize_weights()
  run_training_session(tset1, tset2, tlabels, vset1, vset2, vlabels, conf.NUM_EPOCHS,
                       conv_weights, conv_biases, fc_weights, fc_biases)


def mnist_training_test():
  """
  tests the graph on paired and labeled mnist dataset
  :return: None
  """
  tset1, tset2, tlabels, vset1, vset2, vlabels = dp.get_mnist_dataset()
  conv_weights, conv_biases, fc_weights, fc_biases = initialize_weights()
  run_training_session(tset1, tset2, tlabels, vset1, vset2, vlabels, conf.NUM_EPOCHS,
                       conv_weights, conv_biases, fc_weights, fc_biases,
                       conf.DROP, conf.L2)


if __name__ == "__main__":
  mnist_training_test()
