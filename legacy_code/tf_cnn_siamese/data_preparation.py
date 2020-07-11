import legacy_code.tf_cnn_siamese.configurations as conf
import legacy_code.tf_mnist.generate_datasets as tfdata
import tensorflow as tf
import numpy as np
import fonts.fonts_info as unicodes
import legacy_code.generate_text.generate_character as generator


def training_inputs_placeholders():
  x_1 = tf.placeholder(conf.DTYPE, shape=conf.TRAIN_INPUT_SHAPE, name="x_1")
  x_2 = tf.placeholder(conf.DTYPE, shape=conf.TRAIN_INPUT_SHAPE, name="x_2")
  labels = tf.placeholder(conf.DTYPE, shape=(conf.TRAIN_BATCH_SIZE, 1), name="labels")
  return x_1, x_2, labels


def test_inputs_placeholders():
  x_1 = tf.placeholder(conf.DTYPE, shape=conf.TEST_INPUT_SHAPE, name="x_1")
  x_2 = tf.placeholder(conf.DTYPE, shape=conf.TEST_INPUT_SHAPE, name="x_2")
  labels = tf.placeholder(conf.DTYPE, shape=(conf.TEST_BATCH_SIZE, 1), name="labels")
  return x_1, x_2, labels


def predict_inputs_placeholders():
  x_1 = tf.placeholder(conf.DTYPE, shape=conf.PREDICT_INPUT_SHAPE, name="x_1")
  x_2 = tf.placeholder(conf.DTYPE, shape=conf.PREDICT_INPUT_SHAPE, name="x_2")
  return x_1, x_2


def test_features_placeholders():
  twin_1 = tf.placeholder(conf.DTYPE,
                          shape=(conf.TEST_BATCH_SIZE, conf.NUM_FC_NEURONS),
                          name="twin_1")
  twin_2 = tf.placeholder(conf.DTYPE,
                          shape=(conf.TEST_BATCH_SIZE, conf.NUM_FC_NEURONS),
                          name="twin_2")
  return twin_1, twin_2


def predict_features_placeholders():
  twin_1 = tf.placeholder(conf.DTYPE,
                          shape=(1, conf.NUM_FC_NEURONS),
                          name="twin_1")
  twin_2 = tf.placeholder(conf.DTYPE,
                          shape=(1, conf.NUM_FC_NEURONS),
                          name="twin_2")
  return twin_1, twin_2


def nhwc2nchw(arr):
  transform = [0, 3, 1, 2]
  return np.transpose(arr, transform)


def get_mnist_dataset():
  tset1, tset2, tlabels, vset1, vset2, vlabels = tfdata.compile_transformed_float32_datasets()
  if conf.DATA_FORMAT == 'NCHW':
    tset1 = nhwc2nchw(tset1)
    tset2 = nhwc2nchw(tset2)
    vset1 = nhwc2nchw(vset1)
    vset2 = nhwc2nchw(vset2)
  return tset1, tset2, tlabels, vset1, vset2, vlabels


def generate_normalized_data(num_pairs):
  # pairs of tensors of images with dimention specified in conf
  x_1 = np.reshape(np.random.rand(num_pairs, conf.IMG_X, conf.IMG_Y),
                   (num_pairs, conf.IMG_X, conf.IMG_Y, 1))
  x_2 =  np.reshape(np.random.rand(num_pairs, conf.IMG_X, conf.IMG_Y),
                   (num_pairs, conf.IMG_X, conf.IMG_Y, 1))
  # shifting the range from (0, 1) to (-0.5, 0.5)
  x_1 -= 0.5
  x_2 -= 0.5
  labels = np.random.choice(a=[0, 1], size=num_pairs, p=[0.48, 0.52])
  if conf.DATA_FORMAT == 'NCHW':
    x_1 = nhwc2nchw(x_1)
    x_2 = nhwc2nchw(x_2)
  return x_1, x_2, np.reshape(labels, (num_pairs, 1))


def generate_single_character_image(code_point, font_paths, distorted=True):
  string = chr(code_point)
  img = generator.drawChar(string, conf.FONT_SIZE, font_paths[code_point])
  if distorted:
    img = generator.transformImg(img)
  img = generator.rgb2gray(img)
  return img


def generate_1000k_pairs():
  # actually 1,131,070 pairs
  # 500k look-alikes + 500k random-pairs (for every char, pick another char and have 5 distorted pairs)
  multiplied = 5 * 2
  font_paths = unicodes.map_character_to_single_fontpath()
  implemented = np.where(font_paths != None)
  n = implemented.shape[0] * multiplied
  x_1 = np.empty((n, conf.IMG_X, conf.IMG_Y, conf.NUM_CHANNELS), dtype=np.float_)
  x_2 = np.empty((n, conf.IMG_X, conf.IMG_Y, conf.NUM_CHANNELS), dtype=np.float_)
  y = np.empty(n, dtype=np.float_)
  for i in range(n):
    code_point_1 = implemented[i]
    code_point_2 = np.random.choice(implemented, replace=True)
    while code_point_1 == code_point_2:
      code_point_2 = np.random.choice(implemented, replace=True)
    mismatched = False
    for k in range(multiplied):
      j = i * multiplied + k
      x_1[j] = generate_single_character_image(code_point_1, font_paths)
      if mismatched:
        x_2[j] = generate_single_character_image(code_point_2, font_paths)
        mismatched = True
        y[j] = [0]
      else:
        x_2[j] = generate_single_character_image(code_point_1, font_paths)
        mismatched = False
        y[j] = [1]
  return x_1, x_2, y


def generate_features(num_pairs):
  # pairs of fake features
  twin_1 = np.reshape(np.random.rand(num_pairs, conf.NUM_FC_NEURONS),
                     (num_pairs, conf.NUM_FC_NEURONS))
  twin_2 = np.reshape(np.random.rand(num_pairs, conf.NUM_FC_NEURONS),
                     (num_pairs, conf.NUM_FC_NEURONS))
  return twin_1, twin_2
