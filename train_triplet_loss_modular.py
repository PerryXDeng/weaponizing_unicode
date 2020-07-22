import argparse
import datetime
import numpy as np
import tensorflow as tf
import efficientnet.tfkeras as efn
from sklearn.linear_model import LogisticRegression
from generate_datasets import get_triplet_tf_dataset, get_balanced_pair_tf_dataset
from utilities import allow_gpu_memory_growth, initialize_ckpt_saver, initialize_ckpt_manager, save_checkpoint, \
  restore_checkpoint_if_avail, \
  save_keras_model_weights

parser = argparse.ArgumentParser()
_init_time = datetime.datetime.now()
parser.add_argument('-tri', '--train_iterations', action='store', type=int, default=5000)
parser.add_argument('-bs', '--batch_size', action='store', type=int, default=32)
parser.add_argument('-ts', '--test_sample_size', action='store', type=int, default=4000)
parser.add_argument('-tbs', '--test_batch_size', action='store', type=int, default=24)
parser.add_argument('-dir', '--log_dir', action='store', type=str,
                    default='logs/%s%s' % (_init_time.astimezone().tzinfo.tzname(None),
                                           _init_time.strftime('%Y%m%d_%H_%M_%S_%f')))
parser.add_argument('-ri', '--reporting_interval', action='store', type=int, default=1)
parser.add_argument('-db', '--debug_nan', action='store', type=bool, default=True)
parser.add_argument('-ckpt', '--save_checkpoints', action='store', type=bool, default=False)
# Type of global pooling applied to the output of the last convolutional layer, giving a 2D tensor
# Options: max, avg (None also an option, probably not something we want to use)
parser.add_argument('-p', '--pooling', action='store', type=str, default='avg')
parser.add_argument('-lr', '--learning_rate', action='store', type=float, default=.0005)

# Vector comparison method
# Options: cos, euc
parser.add_argument('-lf', '--loss_function', action='store', type=str, default='cos')
parser.add_argument('-img', '--img_size', action='store', type=int, default=100)
parser.add_argument('-font', '--font_size', action='store', type=float, default=.4)
parser.add_argument('-e', '--epsilon', action='store', type=float, default=10e-5)
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


def train_for_num_step(loss_fn, model, optimizer, triplet_dataset, epsilon, num_steps, debug_nan):
  """

  :param loss_fn: function
  :param model:
  :param optimizer: 
  :param triplet_dataset: 
  :param epsilon:
  :param num_steps:
  :param debug_nan: bool
  :return: mean loss
  """
  loss_sum = 0
  for step, datapoint in enumerate(triplet_dataset):
    if step == num_steps:
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
  return loss_sum / num_steps


def test_for_num_step(measure_fn, model, pairwise_dataset, num_steps, epsilon):
  """
  
  :param measure_fn: distance or similarity function, both works for logistic regression
  :param model: 
  :param pairwise_dataset: 
  :param num_steps: 
  :return: score, accuracy be default
  """
  measures = []
  labs = []
  for step, datapoint in enumerate(pairwise_dataset):
    if step == num_steps:
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


def train():
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
  model = efn.EfficientNetB3(weights='imagenet',
                             input_tensor=tf.keras.layers.Input([args.img_size, args.img_size, 3]), include_top=False,
                             pooling=args.pooling)
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
    mean_loss = train_for_num_step(loss_function, model, optimizer, triplets_dataset, args.epsilon, reporting_interval,
                                   args.debug_nan)
    print(f'Step {i * reporting_interval + 1} Mean Loss: {mean_loss}')
    acc = test_for_num_step(measure_function, model, pairs_dataset, test_iterations, args.epsilon)
    print(f"Step {i * reporting_interval + 1} Testing Acc.: {acc}")
    saver.step.assign_add(reporting_interval)
    if args.save_checkpoints:
      save_checkpoint(ckpt_manager)

  print()

  # serialize model to JSON
  model_json = model.to_json()
  with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
  # serialize weights to HDF5
  model.save_weights("model/model.h5")


if __name__ == '__main__':
  train()
