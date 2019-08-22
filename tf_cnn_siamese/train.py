from tf_cnn_siamese.model import *
import tf_cnn_siamese.data_preparation as dp
import datetime
import time
import sys
import os


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
  x_1, x_2, labels = dp.training_inputs_placeholders()
  optimizer, l = construct_loss_optimizer(x_1, x_2, labels, conv_weights,
                                          conv_biases, fc_weights, fc_biases,
                                          dropout, lagrange)
  # creates session
  saver = tf.train.Saver()
  with tf.Session(config=config) as sess: # automatic tear down of controlled execution
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
    num_steps = total // conf.TRAIN_BATCH_SIZE
    steps_per_epoch = data_size / conf.TRAIN_BATCH_SIZE
    validation_interval = int(conf.EPOCHS_PER_VALIDATION * steps_per_epoch)
    start_time = time.time()
    for step in range(num_steps):
      # offset of the current minibatch
      offset = (step * conf.TRAIN_BATCH_SIZE) % (data_size - conf.TRAIN_BATCH_SIZE)
      batch_x1 = tset1[offset:(offset + conf.TRAIN_BATCH_SIZE), ...]
      batch_x2 = tset2[offset:(offset + conf.TRAIN_BATCH_SIZE), ...]
      batch_labels = ty[offset:(offset + conf.TRAIN_BATCH_SIZE)]
      # maps batched input to graph data nodes
      feed_dict = {x_1: batch_x1, x_2: batch_x2, labels: batch_labels}
      # runs the optimizer every iteration
      sess.run(optimizer, feed_dict=feed_dict)
      # prints loss and validation error when evalidating intermittently
      if step % validation_interval == 0:
        current_epoch = float(step) / steps_per_epoch
        loss = sess.run(l, feed_dict=feed_dict)
        t_accuracy, t_precision, t_recall, t_f1 = batch_validate(tset1, tset2,
                                                 ty, conv_weights, conv_biases,
                                                 fc_weights, fc_biases, sess)
        v_accuracy, v_precision, v_recall, v_f1 = batch_validate(vset1, vset2,
                                                 vy, conv_weights, conv_biases,
                                                 fc_weights, fc_biases, sess)
        elapsed_time = time.time() - start_time
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
  feed_1, feed_2, _ = dp.test_inputs_placeholders()
  model = construct_full_model(feed_1, feed_2, conv_weights, conv_biases,
                               fc_weights, fc_biases)
  size = x1.shape[0]
  if size < conf.TEST_BATCH_SIZE:
    raise ValueError("batch size for validation larger than dataset: %d" % size)
  stats = np.zeros(6)
  for begin in range(0, size, conf.TEST_BATCH_SIZE):
    end = begin + conf.TEST_BATCH_SIZE
    if end <= size:
      stats += calc_stats(session.run(model,
                                      feed_dict={feed_1: x1[begin:end, ...],
                                                 feed_2: x2[begin:end, ...]}),
                                                 labels[begin:end])
    else:
      full_batch_offset = -1 * conf.TEST_BATCH_SIZE
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
  # nvidia rtx 2070 bug workaround
  global config
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  
  mnist_training_test()
