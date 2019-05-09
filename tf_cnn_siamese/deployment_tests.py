from tf_cnn_siamese.model import *
import tf_cnn_siamese.data_preparation as dp
import time
import sys


def naive_predict(x1, x2, path):
  """
  runs the full model (feature extraction and difference metric)
  :param x1: image 1
  :param x2: image 2
  :param path: file path for checkpoint
  :return:
  """
  tf.reset_default_graph()
  conv_weights, conv_biases, fc_weights, fc_biases = initialize_weights()
  saver = tf.train.Saver()
  feed_1, feed_2 = dp.predict_inputs_placeholders()
  out_node = construct_full_model(feed_1, feed_2, conv_weights, conv_biases, fc_weights, fc_biases)
  with tf.Session() as sess: # automatic tear down of controlled execution
    print("\n")
    tf.global_variables_initializer().run()
    print("Data Format " + conf.DATA_FORMAT)
    cuda_enabled = ('NCHW' == conf.DATA_FORMAT)
    print("CUDA Enabled: " + str(cuda_enabled))
    saver.restore(sess, path)
    feed_dict = {feed_1: x1, feed_2:x2}
    out = sess.run(out_node, feed_dict)
  return out


def extract_features(x1, x2, path):
  """
  extracts two set of features from two images
  :param x1: image 1
  :param x2: iamge 2
  :param path: file path for checkpoint
  :return: feature sets for both
  """
  tf.reset_default_graph()
  conv_weights, conv_biases, fc_weights, fc_biases = initialize_weights()
  saver = tf.train.Saver()
  feed_1, feed_2 = dp.predict_inputs_placeholders()
  features_1_node = construct_cnn(feed_1, conv_weights, conv_biases, fc_weights, fc_biases)
  features_2_node = construct_cnn(feed_2, conv_weights, conv_biases, fc_weights, fc_biases)
  with tf.Session() as sess: # automatic tear down of controlled execution
    print("\n")
    tf.global_variables_initializer().run()
    print("Data Format " + conf.DATA_FORMAT)
    cuda_enabled = ('NCHW' == conf.DATA_FORMAT)
    print("CUDA Enabled: " + str(cuda_enabled))
    saver.restore(sess, path)
    feed_dict_1 = {feed_1: x1}
    feed_dict_2 = {feed_2: x2}
    out_1 = sess.run(features_1_node, feed_dict_1)
    out_2 = sess.run(features_2_node, feed_dict_2)
  return out_1, out_2


def predict_with_features(features_1, features_2, path):
  """
  gets model prediction using extracted features
  :param features_1: features from first image
  :param features_2: features from second image
  :param path: file path for checkpoint
  :return: probability of positive
  """
  tf.reset_default_graph()
  _, _, fc_weights, fc_biases = initialize_weights()
  saver = tf.train.Saver()
  feed_1, feed_2 = dp.predict_features_placeholders()
  predict_node = construct_joined_model(feed_1, feed_2, fc_weights, fc_biases)
  with tf.Session() as sess: # automatic tear down of controlled execution
    print("\n")
    tf.global_variables_initializer().run()
    print("Data Format " + conf.DATA_FORMAT)
    cuda_enabled = ('NCHW' == conf.DATA_FORMAT)
    print("CUDA Enabled: " + str(cuda_enabled))
    saver.restore(sess, path)
    feed_dict = {feed_1: features_1, feed_2: features_2}
    out = sess.run(predict_node, feed_dict)
  return out


def test_model_prediction():
  path = input("enter path:")
  tset1, tset2, ty, _, _, _ = dp.get_mnist_dataset()
  x1 = tset1[0:1]
  x2 = tset2[0:1]
  y = ty[0]
  naive = naive_predict(x1, x2, path)
  features_1, features_2 = extract_features(x1, x2, path)
  streamlined = predict_with_features(features_1, features_2, path)
  print("Actual: " + str(y))
  print("Naive: " + str(naive))
  print("Streamlned: " + str(streamlined))


def time_cnn_feature_extraction():
  """
  tests the running time of one twin network doing feature extraction
  :return: None
  """
  tf.reset_default_graph()
  tset1, tset2, ty, vset1, vset2, vy = dp.get_mnist_dataset()
  conv_weights, conv_biases, fc_weights, fc_biases = initialize_weights()
  # creates session
  saver = tf.train.Saver()
  feed_1, feed_2, labels = dp.test_inputs_placeholders()
  cnn = construct_cnn(feed_1, conv_weights, conv_biases, fc_weights, fc_biases)
  with tf.Session() as sess: # automatic tear down of controlled execution
    print("\n")
    tf.global_variables_initializer().run()
    print("Data Format " + conf.DATA_FORMAT)
    cuda_enabled = ('NCHW' == conf.DATA_FORMAT)
    print("CUDA Enabled: " + str(cuda_enabled))
    if input("resume from previous session (y/n):") == "y":
      path = input("enter path:")
      saver.restore(sess, path)
    start_time = time.time()
    # iterates through the training data in batch
    data_size = tset1.shape[0]
    total = data_size
    num_steps = total // conf.TEST_BATCH_SIZE
    for step in range(num_steps):
      # offset of the current minibatch
      offset = (step * conf.TEST_BATCH_SIZE) % (data_size - conf.TEST_BATCH_SIZE)
      batch_x1 = tset1[offset:(offset + conf.TEST_BATCH_SIZE), ...]
      # maps batched input to graph data nodes
      feed_dict = {feed_1: batch_x1}
      # runs the optimizer every iteration
      sess.run(cnn, feed_dict=feed_dict)
      # prints loss and validation error when evalidating intermittently
    elapsed_time = time.time() - start_time
    print('\n%.4f s' %  elapsed_time)
    print("Size: "  + str((ty.shape[0])))
    expected = elapsed_time * 137374 / ty.shape[0]
    print("Expected Feature Extraction Time for 137,374 Unicode: " + str(expected))
    sys.stdout.flush()


def time_joined_difference():
  """
  tests the running time of the joined network running on extracted features
  :return: None
  """
  tf.reset_default_graph()
  num_pairs = 100000
  features_1, features_2 = dp.generate_features(num_pairs)
  _, _, fc_weights, fc_biases = initialize_weights()
  feed_1, feed_2 = dp.test_features_placeholders()
  # creates session
  saver = tf.train.Saver()
  out = construct_joined_model(feed_1, feed_2, fc_weights, fc_biases)
  with tf.Session() as sess: # automatic tear down of controlled execution
    print("\n")
    tf.global_variables_initializer().run()
    print("Data Format " + conf.DATA_FORMAT)
    cuda_enabled = ('NCHW' == conf.DATA_FORMAT)
    print("CUDA Enabled: " + str(cuda_enabled))
    if input("resume from previous session (y/n):") == "y":
      path = input("enter path:")
      saver.restore(sess, path)
    start_time = time.time()
    # iterates through the training data in batch
    total = num_pairs
    num_steps = total // conf.TEST_BATCH_SIZE
    for step in range(num_steps):
      # offset of the current minibatch
      offset = (step * conf.TEST_BATCH_SIZE) % (num_pairs - conf.TEST_BATCH_SIZE)
      batch_x1 = features_1[offset:(offset + conf.TEST_BATCH_SIZE), ...]
      batch_x2 = features_2[offset:(offset + conf.TEST_BATCH_SIZE), ...]
      # maps batched input to graph data nodes
      feed_dict = {feed_1: batch_x1, feed_2: batch_x2}
      # runs the optimizer every iteration
      sess.run(out, feed_dict=feed_dict)
      # prints loss and validation error when evalidating intermittently
    elapsed_time = time.time() - start_time
    print('\n%.4f s' %  elapsed_time)
    print("Size: "  + str(num_pairs))
    expected = elapsed_time * 9435739251 / num_pairs
    print("Expected Feature Extraction Time for 10B Pairs: " + str(expected))
    sys.stdout.flush()


if __name__ == "__main__":
  test_model_prediction()
