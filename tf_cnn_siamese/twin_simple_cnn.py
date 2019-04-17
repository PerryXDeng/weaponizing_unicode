import tf_cnn_siamese.configurations as conf
import tf_cnn_siamese.data_preparation as dp
import tensorflow as tf
import numpy as np
import time
import sys


def single_cnn(x, conv1_weights, conv1_biases, conv2_weights, conv2_biases,
               conv3_weights, conv3_biases, conv4_weights, conv4_biases,
               fc1_weights, fc1_biases, dropout = False):
  conv = tf.nn.conv2d(x,
                      conv1_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='VALID')

  conv = tf.nn.conv2d(pool,
                      conv2_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='VALID')

  conv = tf.nn.conv2d(pool,
                      conv3_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))
  pool = tf.nn.max_pool(relu,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='VALID')
  # last conv layer has no pooling
  conv = tf.nn.conv2d(pool,
                      conv4_weights,
                      strides=[1, 1, 1, 1],
                      padding='SAME')
  relu = tf.nn.relu(tf.nn.bias_add(conv, conv4_biases))

  # Reshape the feature map cuboid into a matrix for fc layers
  features_shape = relu.get_shape().as_list()
  features = tf.reshape(
    pool,
    [features_shape[0], features_shape[1] * features_shape[2] * features_shape[3]])

  # last fc_weights determine output dimensions
  fc = tf.nn.sigmoid(tf.matmul(features, fc1_weights) + fc1_biases)

  # for actual training
  if dropout:
    fc = tf.nn.dropout(fc, conf.DROP_RATE)
  return fc


def construct_logits_model(x_1, x_2, conv1_weights, conv1_biases, conv2_weights,
                           conv2_biases, conv3_weights, conv3_biases, conv4_weights,
                           conv4_biases, fc1_weights, fc1_biases,
                           fcj_weights, fcj_biases, dropout=False):
  # actual neural nets (twin portion)
  twin_1 = single_cnn(x_1, conv1_weights, conv1_biases, conv2_weights,
                      conv2_biases, conv3_weights, conv3_biases, conv4_weights,
                      conv4_biases, fc1_weights, fc1_biases, dropout)
  twin_2 = single_cnn(x_2, conv1_weights, conv1_biases, conv2_weights,
                      conv2_biases, conv3_weights, conv3_biases, conv4_weights,
                      conv4_biases, fc1_weights, fc1_biases, dropout)
  # logits on squared difference (joined portion)
  sq_diff = tf.squared_difference(twin_1, twin_2)
  logits = tf.matmul(sq_diff, fcj_weights) + fcj_biases
  return logits


def construct_full_model(x_1, x_2, conv1_weights, conv1_biases, conv2_weights,
                         conv2_biases, conv3_weights, conv3_biases, conv4_weights,
                         conv4_biases, fc1_weights, fc1_biases, fcj_weights,
                         fcj_biases):
  logits = construct_logits_model(x_1, x_2, conv1_weights, conv1_biases,
                                  conv2_weights, conv2_biases, conv3_weights,
                                  conv3_biases, conv4_weights, conv4_biases,
                                  fc1_weights, fc1_biases, fcj_weights, fcj_biases)
  # actual neural network
  return tf.nn.sigmoid(logits)


def construct_loss_optimizer(x_1, x_2, labels, conv1_weights, conv1_biases,
                             conv2_weights, conv2_biases, conv3_weights, conv3_biases,
                             conv4_weights, conv4_biases,
                             fc1_weights, fc1_biases, fcj_weights, fcj_biases,
                             dropout=False, lagrange=False):
  logits = construct_logits_model(x_1, x_2, conv1_weights, conv1_biases,
                                  conv2_weights, conv2_biases, 
                                  conv3_weights, conv3_biases, conv4_weights,
                                  conv4_biases, fc1_weights,
                                  fc1_biases, fcj_weights, fcj_biases, dropout)
  # cross entropy loss on sigmoids of joined output and labels
  loss_vec = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits)
  loss = tf.reduce_mean(loss_vec)
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
  return trainer, loss, learning_rate


def error_rate(predictions, labels):
  # might be totally incorrect here
  return 1 - (np.sum(np.argmax(predictions, 1) == labels)
              / predictions.shape[0])


def batch_error_rate(set_1, set_2, labels, conv1_weights, conv1_biases,
                     conv2_weights, conv2_biases, conv3_weights, conv3_biases,
                     conv4_weights, conv4_biases, fc1_weights, fc1_biases,
                     fcj_weights, fcj_biases, session):
  x_1, x_2, _ = dp.inputs_placeholders()
  model = construct_full_model(x_1, x_2, conv1_weights, conv1_biases,
                               conv2_weights, conv2_biases, conv3_weights,
                               conv3_biases, conv4_weights, conv4_biases,
                               fc1_weights, fc1_biases, fcj_weights, fcj_biases)
  size = set_1.shape[0]
  if size < conf.BATCH_SIZE:
    raise ValueError("batch size for validation larger than dataset: %d" % size)
  predictions = np.ndarray(shape=size, dtype=np.float32)
  for begin in range(0, size, conf.BATCH_SIZE):
    end = begin + conf.BATCH_SIZE
    if end <= size:
      s1 = set_1[begin:end, ...]
      s2 = set_2[begin:end, ...]
      print(s1.shape)
      print(x_1.shape)
      print(s2.shape)
      print(x_2.shape)
      predictions[begin:end] = session.run(fetches=model,
                                           feed_dict={
                                           x_1: s1,
                                           x_2: s2})
    else:
      batch_predictions = session.run(fetches=model,
                                      feed_dict={
                                      x_1: set_1[-conf.BATCH_SIZE:, ...],
                                      x_2: set_2[-conf.BATCH_SIZE:, ...]})
      predictions[begin:] = batch_predictions[begin - size:, :]
  return error_rate(predictions, labels)


def run_training_session(tset1, tset2, ty, vset1, vset2, vy, epochs,
                         dropout=False, lagrange=False):
  # input nodes
  x_1, x_2, labels = dp.inputs_placeholders()
  # twin portion variables
  # 3x3 filter, depth 64.
  conv1_weights = tf.Variable(tf.truncated_normal([3, 3, 1, 64],
                              stddev=0.1, seed=conf.SEED, dtype=conf.DTYPE),
                              name="twin_conv1_weights")
  conv1_biases = tf.Variable(tf.zeros([64], dtype=conf.DTYPE),
                             name="twin_conv1_biases")
  conv2_weights = tf.Variable(tf.truncated_normal([3, 3, 64, 128],
                              stddev=0.1, seed=conf.SEED, dtype=conf.DTYPE),
                              name="twin_conv2_weights")
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[128], dtype=conf.DTYPE),
                             name="twin_conv2_biases")
  conv3_weights = tf.Variable(tf.truncated_normal([3, 3, 128, 256],
                              stddev=0.1, seed=conf.SEED, dtype=conf.DTYPE),
                              name="twin_conv3_weights")
  conv3_biases = tf.Variable(tf.zeros([256], dtype=conf.DTYPE),
                             name="twin_conv3_biases")
  conv4_weights = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                              stddev=0.1, seed=conf.SEED, dtype=conf.DTYPE),
                              name="twin_conv4_weights")
  conv4_biases = tf.Variable(tf.constant(0.1, shape=[256], dtype=conf.DTYPE),
                             name="twin_conv4_biases")
  # fully connected
  conv_features = 2304 # dims of output from the convlutional layers of twins
  fc_features = 2304 # dims of output from the fc layer of twins
  fc1_weights = tf.Variable(tf.truncated_normal([conv_features, fc_features],
                            stddev=0.1, seed=conf.SEED, dtype=conf.DTYPE),
                            name="twin_fc_weights")
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[fc_features], dtype=conf.DTYPE),
                           name="twin_fc_biases")
  # joined portion variables, turns fc_features into 1 probability
  fcj_weights = tf.Variable(tf.truncated_normal([fc_features, 1], stddev=0.1,
                                                seed=conf.SEED,
                                                dtype=conf.DTYPE),
                            name="joined_fc_weights")
  fcj_biases = tf.Variable(tf.constant(0.1, shape=[1], dtype=conf.DTYPE),
                           name="joined_fc_biases")
  optimizer, l, lr = construct_loss_optimizer(x_1, x_2, labels, conv1_weights,
                                             conv1_biases, conv2_weights,
                                             conv2_biases, conv3_weights,
                                             conv3_biases, conv4_weights,
                                             conv4_biases,fc1_weights,
                                             fc1_biases, fcj_weights, fcj_biases
                                             , dropout, lagrange)
  # creates session
  start_time = time.time()
  with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Initialized!')
    # iterates through the training data in batch
    data_size = tset1.shape[0]
    total = int(epochs * data_size)
    num_steps = total // conf.BATCH_SIZE
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
      if step % conf.VALIDATION_INTERVAL == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        current_epoch = float(step) * conf.BATCH_SIZE / data_size
        # train_error = batch_error_rate(tset1, tset2, ty, conv1_weights,
        #                                conv1_biases, conv2_weights,
        #                                conv2_biases, conv3_weights,
        #                                conv3_biases, conv4_weights,
        #                                conv4_biases, fc1_weights, fc1_biases,
        #                                fcj_weights, fcj_biases, sess)
        # validation_error = batch_error_rate(vset1, vset2, vy, conv1_weights,
        #                                     conv1_biases, conv2_weights,
        #                                     conv2_biases, conv3_weights,
        #                                     conv3_biases, conv4_weights,
        #                                     conv4_biases, fc1_weights,
        #                                     fc1_biases, fcj_weights, fcj_biases,
        #                                     sess)
        print('Step %d (epoch %.2f), %.4f s'
              % (step, current_epoch, elapsed_time))
        loss,learning_rate = sess.run([l,lr], feed_dict=feed_dict)
        print('Minibatch Loss: %.3f, learning rate: %.6f' % (loss, learning_rate))
        # print('Training Error: %.1f%%' % train_error)
        # print('Validation Error: %.1f%%' % validation_error)
        sys.stdout.flush()
    train_error = batch_error_rate(tset1, tset2, ty, conv1_weights,
                                   conv1_biases, conv2_weights,
                                   conv2_biases, conv3_weights,
                                   conv3_biases, conv4_weights,
                                   conv4_biases, fc1_weights, fc1_biases,
                                   fcj_weights, fcj_biases, sess)
    print('Training Error: %.1f%%' % train_error)
    validation_error = batch_error_rate(vset1, vset2, vy, conv1_weights,
                                        conv1_biases, conv2_weights,
                                        conv2_biases, conv3_weights,
                                        conv3_biases, conv4_weights,
                                        conv4_biases, fc1_weights,
                                        fc1_biases, fcj_weights, fcj_biases,
                                        sess)
    print('Validation Error: %.1f%%' % validation_error)


def training_test():
  epochs = 1
  num_pairs = 1000
  tset1, tset2, tlabels = dp.generate_normalized_data(int(0.6 * num_pairs))
  vset1, vset2, vlabels = dp.generate_normalized_data(int(0.2 * num_pairs))
  run_training_session(tset1, tset2, tlabels, vset1, vset2, vlabels, epochs)

if __name__ == "__main__":
  training_test()
