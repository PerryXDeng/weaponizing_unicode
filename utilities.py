import tensorflow as tf
import os


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
