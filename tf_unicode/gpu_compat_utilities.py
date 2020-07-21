import tensorflow as tf


def allow_growth():
  gpus = tf.compat.v1.config.experimental.get_visible_devices('GPU')
  for gpu in gpus:
    try:
      tf.compat.v1.config.experimental.set_memory_growth(gpu, True)
    except:
      pass

