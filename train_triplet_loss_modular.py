import argparse
import datetime
import os
import sys
import time
import math
import numpy as np
import tensorflow as tf
import efficientnet.tfkeras as efn
from sklearn.linear_model import LogisticRegression
from generate_datasets import get_triplet_tf_dataset, get_balanced_pair_tf_dataset
from utilities import allow_gpu_memory_growth, initialize_ckpt_saver, initialize_ckpt_manager, save_checkpoint, \
  restore_checkpoint_if_avail
from absl import logging

parser = argparse.ArgumentParser()
_init_time = datetime.datetime.now()
# BS * 7
parser.add_argument('-tri', '--train_iterations', action='store', type=int, default=5000)
parser.add_argument('-trs', '--train_seconds', action='store', type=int, default=60*5)
parser.add_argument('-bs', '--batch_size', action='store', type=int, default=32)
parser.add_argument('-bm', '--batch_multiplier', action='store', type=int, default=5)
parser.add_argument('-ts', '--test_sample_size', action='store', type=int, default=4000)
parser.add_argument('-tbs', '--test_batch_size', action='store', type=int, default=24)
parser.add_argument('-dir', '--log_dir', action='store', type=str,
                    default='logs/%s%s' % (_init_time.astimezone().tzinfo.tzname(None),
                                           _init_time.strftime('%Y%m%d_%H_%M_%S_%f')))
parser.add_argument('-ri', '--reporting_interval', action='store', type=int, default=5)
parser.add_argument('-ris', '--reporting_interval_seconds', action='store', type=int, default=10)
parser.add_argument('-db', '--debug_nan', action='store', type=bool, default=False)
parser.add_argument('-ckpt', '--save_checkpoints', action='store', type=bool, default=False)
# Type of global pooling applied to the output of the last convolutional layer, giving a 2D tensor
# Options: max, avg (None also an option, probably not something we want to use)
parser.add_argument('-p', '--pooling', action='store', type=str, default='avg')
parser.add_argument('-lr', '--learning_rate', action='store', type=float, default=.0005)

# Vector comparison method
# Options: cos, euc
parser.add_argument('-lf', '--loss_function', action='store', type=str, default='cos')
parser.add_argument('-img', '--img_size', action='store', type=int, default=100)
parser.add_argument('-font', '--font_size', action='store', type=float, default=.5)
parser.add_argument('-e', '--epsilon', action='store', type=float, default=10e-5)
parser.add_argument('-m', '--efn_model', action='store', type=str, default='B3')
parser.add_argument('-dcr', '--drop_connect_rate', action='store', type=float, default=0.2)
parser.add_argument('-t', '--tune', action='store', type=bool, default=False)
parser.add_argument('-sm', '--save_model', action='store', type=bool, default=False)
parser.add_argument('-fdp', '--font_dict_path', action='store', type=str, default="./fonts/multifont_mapping.pkl")
args = parser.parse_args()


@tf.function
def cos_sim(x1, x2, epsilon):
  axis_1 = x1.shape[0]
  axis_2 = x2.shape[1]
  a_v = tf.reshape(x1, [axis_1, 1, axis_2])
  b_v = tf.reshape(x2, [axis_1, axis_2, 1])
  return tf.reshape(tf.matmul(a_v, b_v), [axis_1]) / ((tf.norm(x1, axis=1) * tf.norm(x2, axis=1)) + epsilon)


@tf.function
# Where x1 is an anchor input, x2 belongs to the same class and x3 belongs to a different class
def cos_triplet_loss(anc, pos, neg, epsilon):
  return (tf.reduce_mean((cos_sim(anc, neg, epsilon) - cos_sim(anc, pos, epsilon))) + 2) / 4


@tf.function
# Where x1 is an anchor input, x2 belongs to the same class and x3 belongs to a different class
def euc_triplet_loss(anc, pos, neg, epsilon):
  return tf.reduce_mean(tf.norm(anc - pos, axis=1) - tf.norm(anc - neg, axis=1))


@tf.function
def floatify_and_normalize(data):
  data = tf.cast(data, tf.float32)
  return (data - (255 / 2)) / (255 / 2)


def train_for_num_batch(loss_fn, model, optimizer, triplet_dataset, epsilon, num_batch, debug_nan):
  """

  :param loss_fn: function
  :param model:
  :param optimizer: 
  :param triplet_dataset: 
  :param epsilon:
  :param num_batch:
  :param debug_nan: bool
  :return: mean loss
  """
  loss_sum = 0
  for step, datapoint in enumerate(triplet_dataset):
    if step == num_batch:
      break
    with tf.GradientTape() as tape:
      anc, pos, neg = datapoint
      anchor_forward, positive_forward, negative_forward = model(anc), model(pos), model(neg)
      loss = loss_fn(anchor_forward, positive_forward, negative_forward, epsilon)
      loss_sum += loss.numpy()
    # Get gradients of loss wrt the weights.
    gradients = tape.gradient(loss, model.trainable_weights)
    # Update the weights of the model.
    grad_check = None
    if debug_nan:
      grad_check = [tf.debugging.check_numerics(g, message='Gradient NaN Found!!!') for g in gradients if
                    g is not None] + [tf.debugging.check_numerics(loss, message="Loss NaN Found!!!")]
    with tf.control_dependencies(grad_check):
      optimizer.apply_gradients(zip(gradients, model.trainable_weights))
  return loss_sum / num_batch


def train_for_num_minibatch(loss_fn, model, optimizer, triplet_dataset, epsilon, total_minibatch, batch_multiplier):
  """

  :param loss_fn: function
  :param model:
  :param optimizer:
  :param triplet_dataset:
  :param epsilon:
  :param total_minibatch:
  :param batch_multiplier:
  :return: mean loss
  """
  loss_sum = 0
  num_batch = math.ceil(total_minibatch / batch_multiplier)
  total_minibatch_count = 0
  for _ in range(num_batch):
    mini_grads = []
    minibatch_count = 0
    for datapoint in triplet_dataset:
      if minibatch_count == batch_multiplier or total_minibatch_count == total_minibatch:
        break
      with tf.GradientTape() as tape:
        anc, pos, neg = datapoint
        anchor_forward, positive_forward, negative_forward = model(anc), model(pos), model(neg)
        loss = loss_fn(anchor_forward, positive_forward, negative_forward, epsilon)
        loss_sum += loss.numpy()
      # Get gradients of loss wrt the weights.
      mini_grads.append(zip(tape.gradient(loss, model.trainable_weights), model.trainable_weights))
      minibatch_count += 1
      total_minibatch_count += 1
    average_grads = []
    for grad_and_vars in zip(*mini_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_minibatch0, var0_gpu0), ... , (grad0_minibatchN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)

      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    optimizer.apply_gradients(average_grads)
  return loss_sum / total_minibatch


def test_for_num_minibatch(measure_fn, model, pairwise_dataset, num_minibatch, epsilon):
  """
  
  :param measure_fn: distance or similarity function, both works for logistic regression
  :param model: 
  :param pairwise_dataset: 
  :param num_minibatch: 
  :return: score, accuracy be default
  """
  measures = []
  labs = []
  for step, datapoint in enumerate(pairwise_dataset):
    if step == num_minibatch:
      break
    batch_x1, batch_x2, batch_y = datapoint
    batch_measures = measure_fn(model.predict(batch_x1), model.predict(batch_x2), epsilon)
    measures.append(batch_measures.numpy())
    labs.append(batch_y.numpy())
  measures = np.stack(measures).reshape([-1, 1])
  labs = np.stack(labs).reshape([-1])
  # no regularization, no randomization
  regression_fitter = LogisticRegression(penalty='none', random_state=0, solver='saga')
  regression_fitter.fit(measures, labs)
  return regression_fitter.score(measures, labs)
  
def get_efn_model(model_version,img_size,pooling, drop_connect_rate):
  if model_version == 'B2':
    return efn.EfficientNetB2(weights='imagenet',
                             input_tensor=tf.keras.layers.Input([img_size, img_size, 3]), include_top=False,
                             pooling=pooling, drop_connect_rate=drop_connect_rate)
  elif model_version == 'B3':
    return efn.EfficientNetB3(weights='imagenet',
                             input_tensor=tf.keras.layers.Input([img_size, img_size, 3]), include_top=False,
                             pooling=pooling, drop_connect_rate=drop_connect_rate)
  elif model_version == 'B4':
    return efn.EfficientNetB4(weights='imagenet',
                             input_tensor=tf.keras.layers.Input([img_size, img_size, 3]), include_top=False,
                             pooling=pooling, drop_connect_rate=drop_connect_rate)
  elif model_version == 'B5':
    return efn.EfficientNetB5(weights='imagenet',
                             input_tensor=tf.keras.layers.Input([img_size, img_size, 3]), include_top=False,
                             pooling=pooling, drop_connect_rate=drop_connect_rate)


def train_for_num_seconds(loss_fn, model, optimizer, triplet_dataset, epsilon, num_seconds, debug_nan):
  """

  :param loss_fn: function
  :param model:
  :param optimizer:
  :param triplet_dataset:
  :param epsilon:
  :param num_seconds:
  :param debug_nan: bool
  :return: mean loss
  """
  loss_sum = 0
  start_time = time.time()
  num_minibatch = 0
  for datapoint in triplet_dataset:
    if time.time() - start_time > num_seconds:
      break
    with tf.GradientTape() as tape:
      anc, pos, neg = datapoint
      anchor_forward, positive_forward, negative_forward = model(anc), model(pos), model(neg)
      loss = loss_fn(anchor_forward, positive_forward, negative_forward, epsilon)
      loss_sum += loss.numpy()
    # Get gradients of loss wrt the weights.
    gradients = tape.gradient(loss, model.trainable_weights)
    # Update the weights of the model.
    grad_check = None
    if debug_nan:
      grad_check = [tf.debugging.check_numerics(g, message='Gradient NaN Found!!!') for g in gradients if
                    g is not None] + [tf.debugging.check_numerics(loss, message="Loss NaN Found!!!")]
    with tf.control_dependencies(grad_check):
      optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    num_minibatch += 1
  return loss_sum / num_minibatch


def train_time():
  logging.set_verbosity(logging.INFO)
  allow_gpu_memory_growth()
  if args.loss_function == 'cos':
    loss_function = cos_triplet_loss
    measure_function = cos_sim
  elif args.loss_function == 'euc':
    loss_function = euc_triplet_loss
    measure_function = lambda a, b, epsilon: tf.norm(a - b, axis=1)
  else:
    loss_function = None
    measure_function = None
  model = get_efn_model(args.efn_model,args.img_size,args.pooling,args.drop_connect_rate)
  # Training Settings
  optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

  saver = initialize_ckpt_saver(model, optimizer)
  ckpt_manager = initialize_ckpt_manager(saver, args.log_dir)

  preprocess_triplets = lambda x, y, z: (
      floatify_and_normalize(x), floatify_and_normalize(y), floatify_and_normalize(z))
  preprocess_pairs = lambda x, y, z: (floatify_and_normalize(x), floatify_and_normalize(y), z)

  triplets_dataset = get_triplet_tf_dataset(args.img_size, args.font_size, preprocess_fn=preprocess_triplets,
                                            batch_size=args.batch_size, path_prefix="./fonts/")
  pairs_dataset = get_balanced_pair_tf_dataset(args.img_size, args.font_size, batch_size=args.test_batch_size,
                                               preprocess_fn=preprocess_pairs, path_prefix="./fonts/")

  # Training Loop
  reporting_interval = args.reporting_interval_seconds
  test_iterations = args.test_sample_size // args.test_batch_size
  restore_checkpoint_if_avail(saver, ckpt_manager)
  start_time = time.time()
  iteration = 0
  elapsed_seconds = lambda : time.time() - start_time
  readable_time = lambda secs : str(datetime.timedelta(seconds=secs))
  while (iteration * reporting_interval) < args.train_seconds:
    mean_loss = train_for_num_seconds(loss_function, model, optimizer, triplets_dataset, args.epsilon, reporting_interval,
                                      args.debug_nan)
    acc = test_for_num_minibatch(measure_function, model, pairs_dataset, test_iterations, args.epsilon)
    saver.step.assign_add(reporting_interval)
    if args.save_checkpoints:
      save_checkpoint(ckpt_manager)
    iteration += 1
    logging.info(f'Iteration {iteration}')
    logging.info(f"Elapsed Time.: {readable_time(elapsed_seconds())}")
    logging.info(f'Elapsed Training Time: {readable_time(iteration * reporting_interval)}')
    logging.info(f'Mean Loss: {mean_loss}')
    logging.info(f"Testing Acc.: {acc}")

  # serialize model to JSON
  model_json = model.to_json()
  with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model/model.h5")


def train_steps():
  logging.set_verbosity(logging.INFO)
  allow_gpu_memory_growth()
  if args.loss_function == 'cos':
    loss_function = cos_triplet_loss
    measure_function = cos_sim
  elif args.loss_function == 'euc':
    loss_function = euc_triplet_loss
    measure_function = lambda a, b, epsilon: tf.norm(a - b, axis=1)
  else:
    loss_function = None
    measure_function = None
  model = get_efn_model(args.efn_model,args.img_size,args.pooling,args.drop_connect_rate)
  # Training Settings
  optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

  saver = initialize_ckpt_saver(model, optimizer)
  ckpt_manager = initialize_ckpt_manager(saver, args.log_dir)

  preprocess_triplets = lambda x, y, z: (
      floatify_and_normalize(x), floatify_and_normalize(y), floatify_and_normalize(z))
  preprocess_pairs = lambda x, y, z: (floatify_and_normalize(x), floatify_and_normalize(y), z)

  triplets_dataset = get_triplet_tf_dataset(args.img_size, args.font_size, preprocess_fn=preprocess_triplets,
                                            batch_size=args.batch_size)
  pairs_dataset = get_balanced_pair_tf_dataset(args.img_size, args.font_size, batch_size=args.test_batch_size,
                                               preprocess_fn=preprocess_pairs)

  # Training Loop
  reporting_interval = args.reporting_interval
  test_iterations = args.test_sample_size // args.test_batch_size
  restore_checkpoint_if_avail(saver, ckpt_manager)
  for i in range(args.train_iterations // reporting_interval):
    mean_loss = train_for_num_batch(loss_function, model, optimizer, triplets_dataset, args.epsilon, reporting_interval,
                                    args.debug_nan)
    logging.info(f'Batch {i * reporting_interval + 1}')
    logging.info(f'Mean Loss: {mean_loss}')
    acc = test_for_num_minibatch(measure_function, model, pairs_dataset, test_iterations, args.epsilon)
    logging.info(f"Testing Acc.: {acc}")
    saver.step.assign_add(reporting_interval)
    if args.save_checkpoints:
      save_checkpoint(ckpt_manager)

  # serialize model to JSON
  model_json = model.to_json()
  with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model/model.h5")


def train_steps_minibatch():
  logging.set_verbosity(logging.INFO)
  allow_gpu_memory_growth()
  if args.loss_function == 'cos':
    loss_function = cos_triplet_loss
    measure_function = cos_sim
  elif args.loss_function == 'euc':
    loss_function = euc_triplet_loss
    measure_function = lambda a, b, epsilon: tf.norm(a - b, axis=1)
  else:
    loss_function = None
    measure_function = None
  model = get_efn_model(args.efn_model,args.img_size,args.pooling,args.drop_connect_rate)
  # Training Settings
  optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

  saver = initialize_ckpt_saver(model, optimizer)
  ckpt_manager = initialize_ckpt_manager(saver, args.log_dir)

  preprocess_triplets = lambda x, y, z: (
      floatify_and_normalize(x), floatify_and_normalize(y), floatify_and_normalize(z))
  preprocess_pairs = lambda x, y, z: (floatify_and_normalize(x), floatify_and_normalize(y), z)

  triplets_dataset = get_triplet_tf_dataset(args.img_size, args.font_size, preprocess_fn=preprocess_triplets,
                                            batch_size=args.batch_size)
  pairs_dataset = get_balanced_pair_tf_dataset(args.img_size, args.font_size, batch_size=args.test_batch_size,
                                               preprocess_fn=preprocess_pairs)

  # Training Loop
  reporting_interval = args.reporting_interval
  test_iterations = args.test_sample_size // args.test_batch_size
  restore_checkpoint_if_avail(saver, ckpt_manager)
  for i in range(args.train_iterations // reporting_interval):
    mean_loss = train_for_num_minibatch(loss_function, model, optimizer, triplets_dataset, args.epsilon, reporting_interval, args.batch_multiplier)
    logging.info(f'Minibatch {(i+1) * reporting_interval}')
    logging.info(f'Mean Loss: {mean_loss}')
    acc = test_for_num_minibatch(measure_function, model, pairs_dataset, test_iterations, args.epsilon)
    logging.info(f"Testing Acc.: {acc}")
    saver.step.assign_add(reporting_interval)
    if args.save_checkpoints:
      save_checkpoint(ckpt_manager)

  # serialize model to JSON
  model_json = model.to_json()
  with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model/model.h5")


def train_tune_cli():
  logging.set_verbosity(logging.INFO)
  allow_gpu_memory_growth()
  if args.tune:
    if not os.path.exists(args.log_dir):
      os.makedirs(args.log_dir)
    sys.stdout = open(os.path.join(args.log_dir, "stdout.txt"), 'a')
  if args.loss_function == 'cos':
    loss_function = cos_triplet_loss
    measure_function = cos_sim
  elif args.loss_function == 'euc':
    loss_function = euc_triplet_loss
    measure_function = lambda a, b, epsilon: tf.norm(a - b, axis=1)
  else:
    loss_function = None
    measure_function = None
  model = get_efn_model(args.efn_model,args.img_size,args.pooling,args.drop_connect_rate)
  # Training Settings
  optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

  saver = initialize_ckpt_saver(model, optimizer)
  ckpt_manager = initialize_ckpt_manager(saver, args.log_dir)

  preprocess_triplets = lambda x, y, z: (
      floatify_and_normalize(x), floatify_and_normalize(y), floatify_and_normalize(z))
  preprocess_pairs = lambda x, y, z: (floatify_and_normalize(x), floatify_and_normalize(y), z)

  triplets_dataset = get_triplet_tf_dataset(args.img_size, args.font_size, preprocess_fn=preprocess_triplets,
                                            batch_size=args.batch_size,font_dict_path=args.font_dict_path, path_prefix='../../fonts/')
  pairs_dataset = get_balanced_pair_tf_dataset(args.img_size, args.font_size, batch_size=args.test_batch_size,
                                               preprocess_fn=preprocess_pairs,font_dict_path=args.font_dict_path, path_prefix='../../fonts/')

  # Training Loop
  reporting_interval = args.reporting_interval
  test_iterations = args.test_sample_size // args.test_batch_size
  restore_checkpoint_if_avail(saver, ckpt_manager)
  final_training_acc = 0
  for i in range(args.train_iterations // reporting_interval):
    mean_loss = train_for_num_batch(loss_function, model, optimizer, triplets_dataset, args.epsilon, reporting_interval,
                                    args.debug_nan)
    logging.info(f'Batch {i * reporting_interval + 1} Mean Loss: {mean_loss}')
    acc = test_for_num_minibatch(measure_function, model, pairs_dataset, test_iterations, args.epsilon)
    final_training_acc = acc
    logging.info(f"Batch {i * reporting_interval + 1} Testing Acc.: {acc}")
    saver.step.assign_add(reporting_interval)
    if args.save_checkpoints:
      save_checkpoint(ckpt_manager)

  if args.save_model:
    # serialize model to JSON
    model_json = model.to_json()
    with open("model/model.json", "w") as json_file:
      json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/model.h5")
  if args.tune:
    textfile = open(f"{args.log_dir}/metric.txt", 'a')
    textfile.write(f'{final_training_acc}\n')
    textfile.close()

if __name__ == '__main__':
  if args.tune:
    train_tune_cli()
  else:
    train_steps_minibatch()
