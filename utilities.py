import tensorflow as tf
import os


class ModifiedL2Regularization:
  """
  a custom regularization method for transfer learning
  """
  def __init__(self, fresh_model, multiplier):
    if multiplier > 0:
      weights = [tf.reshape(tf.stop_gradient(tf.identity(w)), [-1]) for w in fresh_model.trainable_weights]
      self.weight_size = [tf.shape(w)[0].numpy() for w in weights]
      self.ragged_fresh_weights = tf.RaggedTensor.from_row_lengths(tf.concat(weights, axis=0),
                                                                   row_lengths=self.weight_size)
      self.multiplier = multiplier
    else:
      self.multiplier = None

  @tf.function
  def loss(self, trained_model):
    if self.multiplier:
      ragged_trained_weights = tf.RaggedTensor.from_row_lengths(
        tf.concat([tf.reshape(v, [-1]) for v in trained_model.trainable_weights], axis=0),
        row_lengths=self.weight_size)
      return self.multiplier * tf.math.reduce_sum(tf.math.squared_difference(self.ragged_fresh_weights, ragged_trained_weights))
    else:
      with tf.device('/device:GPU:0'):
        return tf.constant(0.0)

  def __call__(self, trained_model):
    return self.loss(trained_model)

def allow_gpu_memory_growth():
  gpus = tf.compat.v1.config.experimental.get_visible_devices('GPU')
  for gpu in gpus:
    try:
      tf.compat.v1.config.experimental.set_memory_growth(gpu, True)
    except:
      pass


def initialize_ckpt_saver(model, optimizer):
  return tf.train.Checkpoint(step=tf.Variable(1), net=model, optimizer=optimizer)


def initialize_ckpt_manager(saver, save_dir):
  return tf.train.CheckpointManager(saver, save_dir, max_to_keep=3)


def save_checkpoint(ckpt_manager):
  # saves both the optimizer and model state
  ckpt_manager.save()


def restore_checkpoint_if_avail(saver, ckpt_manager):
  # restores the latest ckpt if there is one
  if ckpt_manager.latest_checkpoint:
    saver.restore(ckpt_manager.latest_checkpoint)


def load_keras_model_weights(path):
  return tf.keras.models.load_model(path)


def save_keras_model_weights(model, save_dir):
  # serialize weights to HDF5
  # does not save optimizer state
  model.save_weights(os.path.join(save_dir, "model.h5"))
