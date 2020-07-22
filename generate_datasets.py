from generate_character import transformImg, drawChar
from collections import Iterator
import matplotlib.pyplot as plt
import pickle
import numpy as np
import random
import cv2 as cv
import tensorflow as tf

FONTS_PATH = './fonts/'
DRAW_ATTEMPTS = 25


# TODO: check for replacement drawing i.e. 63609 63656 63705 63582 63606 63618
def try_draw_char(char, available_fonts, empty_image, img_size, font_size):
  if len(available_fonts) == 0:
    # No fonts support drawing this unicode character, or the unicode character is corrupt!
    return empty_image
  else:
    selected_font = available_fonts[random.randint(0, len(available_fonts) - 1)]
    try:
      img = transformImg(drawChar(img_size, chr(char), font_size, FONTS_PATH + selected_font))
    except ValueError:
      # The drawn character is to big for the desired region!
      available_fonts.remove(selected_font)
      return try_draw_char(char, available_fonts, empty_image, img_size, font_size)
    if (img == empty_image).all():
      # The selected font does not support drawing this character!
      available_fonts.remove(selected_font)
      return try_draw_char(char, available_fonts, empty_image, img_size, font_size)
    else:
      return img


def compile_datasets(training_size, test_size, font_size=.2, img_size=200, color_format='gray'):
  empty_image = np.full((img_size, img_size), 255)
  infile = open(FONTS_PATH + 'multifont_mapping.pkl', 'rb')
  unicode_mapping_dict = pickle.load(infile)
  infile.close()  # Perry: it's programming best practice to close such file pointers, or use Python with syntax
  # TODO!!: calculate number of supported codepoints, important for clustering optimization and paper
  # Return dictionary of unsupported characters
  unicode_count = len(unicode_mapping_dict)
  infile.close()
  unicode_chars_available = list(unicode_mapping_dict.keys())
  unicode_chars_population = list(unicode_mapping_dict.keys())
  if color_format == 'RGB':
    train_img_shape = (training_size, img_size, img_size, 3)
    test_img_shape = (test_size, img_size, img_size, 3)
  else:
    train_img_shape = (training_size, img_size, img_size)
    test_img_shape = (test_size, img_size, img_size)
  anchors = np.empty(train_img_shape, dtype=np.uint8)
  positives = np.empty(train_img_shape, dtype=np.uint8)
  negatives = np.empty(train_img_shape, dtype=np.uint8)
  x1_test = np.empty(test_img_shape, dtype=np.uint8)
  x2_test = np.empty(test_img_shape, dtype=np.uint8)
  y_test = np.arange(test_size, dtype=np.uint8) % 2
  for i in range(training_size):
    anchor_img = empty_image
    negative_img = empty_image
    positive_img = empty_image
    while (anchor_img == empty_image).all():
      anchor_char = unicode_chars_available[random.randint(0, len(unicode_chars_available) - 1)]
      unicode_chars_available.remove(anchor_char)
      supported_anchor_fonts = unicode_mapping_dict[anchor_char]
      anchor_img = try_draw_char(anchor_char, supported_anchor_fonts, empty_image, img_size, font_size)
    while (negative_img == empty_image).all():
      negative_char = anchor_char
      while negative_char == anchor_char:
        negative_char = unicode_chars_population[random.randint(0, unicode_count - 1)]
      supported_negative_fonts = unicode_mapping_dict[negative_char]
      negative_img = try_draw_char(negative_char, supported_negative_fonts, empty_image, img_size, font_size)
    draw_attempts = 0
    while (positive_img == empty_image).all():
      # Possible fonts need to be regenerated because the drawing function is bugged
      supported_positive_fonts = unicode_mapping_dict[anchor_char]
      # print(anchor_char, len(supported_positive_fonts))
      positive_img = try_draw_char(anchor_char, supported_positive_fonts, empty_image, img_size, font_size)
      draw_attempts += 1
      if draw_attempts > DRAW_ATTEMPTS:
        positive_img = transformImg(anchor_img)
        break
    if color_format == 'RGB':
      anchor_img = cv.cvtColor(anchor_img, cv.COLOR_GRAY2RGB)
      negative_img = cv.cvtColor(negative_img, cv.COLOR_GRAY2RGB)
      positive_img = cv.cvtColor(positive_img, cv.COLOR_GRAY2RGB)
    anchors[i] = anchor_img
    negatives[i] = negative_img
    positives[i] = positive_img
  for i in range(test_size):
    x1_test_img = empty_image
    x2_test_img = empty_image
    while (x1_test_img == empty_image).all():
      x1_char = unicode_chars_available[random.randint(0, len(unicode_chars_available) - 1)]
      unicode_chars_available.remove(x1_char)
      supported_x1_fonts = unicode_mapping_dict[x1_char]
      x1_test_img = try_draw_char(x1_char, supported_x1_fonts, empty_image, img_size, font_size)
    if y_test[i] == 1:
      draw_attempts = 0
      while (x2_test_img == empty_image).all():
        # Possible fonts need to be regenerated because the drawing function is bugged
        supported_x2_fonts = unicode_mapping_dict[x1_char]
        x2_test_img = try_draw_char(x1_char, supported_x2_fonts, empty_image, img_size, font_size)
        draw_attempts += 1
        if draw_attempts > DRAW_ATTEMPTS:
          x2_test_img = transformImg(x1_test_img)
          break
    else:
      while (x2_test_img == empty_image).all():
        x2_char = x1_char
        while x2_char == x1_char:
          x2_char = unicode_chars_population[random.randint(0, unicode_count - 1)]
        supported_x2_fonts = unicode_mapping_dict[x2_char]
        x2_test_img = try_draw_char(x2_char, supported_x2_fonts, empty_image, img_size, font_size)
    if color_format == 'RGB':
      x1_test_img = cv.cvtColor(x1_test_img, cv.COLOR_GRAY2RGB)
      x2_test_img = cv.cvtColor(x2_test_img, cv.COLOR_GRAY2RGB)
    x1_test[i] = x1_test_img
    x2_test[i] = x2_test_img
  return anchors, positives, negatives, x1_test, x2_test, y_test


class AbstractUnicodeRendererIterable(Iterator):

  def __init__(self, img_size: int, font_size: float, font_dict_path: str = "./fonts/multifont_mapping.pkl",
               rgb: bool = True):
    self.img_size = img_size
    self.empty_image = np.full((img_size, img_size), 255, dtype=np.uint8)
    with open(font_dict_path, 'rb') as fp:
      self.unicode_mapping_dict = pickle.load(fp)
    self.codepoints = list(self.unicode_mapping_dict.keys())
    self.num_codepoints = len(self.codepoints)
    self.font_size = font_size
    self.draw_with_replacement = lambda: self.codepoints[random.randint(0, self.num_codepoints - 1)]
    self.rgb = rgb

  def __iter__(self):
    return self

  def __next__(self):
    raise NotImplementedError


class TripletIterable(AbstractUnicodeRendererIterable):

  def __next__(self):
    anchor_img = self.empty_image
    negative_img = self.empty_image
    positive_img = self.empty_image
    while (anchor_img == self.empty_image).all():
      anchor_char = self.draw_with_replacement()
      supported_anchor_fonts = self.unicode_mapping_dict[anchor_char]
      anchor_img = try_draw_char(anchor_char, supported_anchor_fonts, self.empty_image,
                                 self.img_size, self.font_size)
    while (negative_img == self.empty_image).all():
      negative_char = anchor_char
      while negative_char == anchor_char:
        negative_char = self.draw_with_replacement()
      supported_negative_fonts = self.unicode_mapping_dict[negative_char]
      negative_img = try_draw_char(negative_char, supported_negative_fonts, self.empty_image,
                                   self.img_size, self.font_size)
    while (positive_img == self.empty_image).all():
      # Possible fonts need to be regenerated because the drawing function is bugged
      supported_positive_fonts = self.unicode_mapping_dict[anchor_char]
      # print(anchor_char, len(supported_positive_fonts))
      positive_img = try_draw_char(anchor_char, supported_positive_fonts, self.empty_image,
                                   self.img_size, self.font_size)
    if self.rgb:
      anchor_img = cv.cvtColor(anchor_img, cv.COLOR_GRAY2RGB)
      negative_img = cv.cvtColor(negative_img, cv.COLOR_GRAY2RGB)
      positive_img = cv.cvtColor(positive_img, cv.COLOR_GRAY2RGB)
    return anchor_img, positive_img, negative_img


class BalancedPairIterable(AbstractUnicodeRendererIterable):

  def __init__(self, img_size: int, font_size: float, font_dict_path: str = "./fonts/multifont_mapping.pkl",
               rgb: bool = True, p_neg: float = 0.5):
    super().__init__(img_size, font_size, font_dict_path, rgb)
    self.p_neg = p_neg

  def __next__(self):
    img_a = self.empty_image
    img_b = self.empty_image
    lab = 1.0
    while (img_a == self.empty_image).all():
      codepoint_a = self.draw_with_replacement()
      supported_a_fonts = self.unicode_mapping_dict[codepoint_a]
      img_a = try_draw_char(codepoint_a, supported_a_fonts, self.empty_image,
                            self.img_size, self.font_size)
    while (img_b == self.empty_image).all():
      codepoint_b = codepoint_a
      if random.random() < self.p_neg:
        lab = 0.0
        while codepoint_b == codepoint_a:
          codepoint_b = self.draw_with_replacement()
      supported_b_fonts = self.unicode_mapping_dict[codepoint_b]
      img_b = try_draw_char(codepoint_b, supported_b_fonts, self.empty_image,
                            self.img_size, self.font_size)
    if self.rgb:
      img_a = cv.cvtColor(img_a, cv.COLOR_GRAY2RGB)
      img_b = cv.cvtColor(img_b, cv.COLOR_GRAY2RGB)
    return img_a, img_b, lab


@tf.autograph.experimental.do_not_convert
def get_triplet_tf_dataset(img_size: int, font_size: float, font_dict_path: str = "./fonts/multifont_mapping.pkl",
                           rgb: bool = True,
                           num_workers: int = 1, preprocess_fn=lambda x, y, z: (x, y, z), batch_size: int = 2,
                           buffer_size: int = 4):
  """

  :param img_size:
  :param font_size:
  :param fonts_path:
  :param rgb:
  :param num_workers:
  :param preprocess_fn:
  :param batch_size:
  :param buffer_size:
  :return: dataset iterable
  """
  if rgb:
    img_shape = [img_size, img_size, 3]
  else:
    img_shape = [img_size, img_size, 1]
  return tf.compat.v1.data.Dataset.from_generator(TripletIterable, args=(img_size, font_size, font_dict_path, rgb),
                                                  output_types=(tf.uint8, tf.uint8, tf.uint8),
                                                  output_shapes=(tf.TensorShape(img_shape),
                                                                 tf.TensorShape(img_shape),
                                                                 tf.TensorShape(img_shape))) \
    .map(preprocess_fn, num_parallel_calls=num_workers).batch(batch_size, drop_remainder=True).prefetch(buffer_size)


@tf.autograph.experimental.do_not_convert
def get_balanced_pair_tf_dataset(img_size: int, font_size: float, font_dict_path: str = "./fonts/multifont_mapping.pkl",
                                 rgb: bool = True,
                                 num_workers: int = 1, preprocess_fn=lambda x, y, z: (x, y, z), batch_size: int = 2,
                                 buffer_size: int = 4):
  """

  :param img_size:
  :param font_size:
  :param font_dict_path:
  :param rgb:
  :param num_workers:
  :param preprocess_fn:
  :param batch_size:
  :param buffer_size:
  :return: dataset iterable
  """
  if rgb:
    img_shape = [img_size, img_size, 3]
  else:
    img_shape = [img_size, img_size, 1]
  return tf.compat.v1.data.Dataset.from_generator(BalancedPairIterable,
                                                  args=(img_size, font_size, font_dict_path, rgb),
                                                  output_types=(tf.uint8, tf.uint8, tf.float32),
                                                  output_shapes=(tf.TensorShape(img_shape),
                                                                 tf.TensorShape(img_shape),
                                                                 tf.TensorShape([]))) \
    .map(preprocess_fn, num_parallel_calls=num_workers).batch(batch_size, drop_remainder=True).prefetch(buffer_size)


def test_drawing(font_size=.2, img_size=200):
  # 17 corrupt unicode chars
  # 3600 invalid pixel size
  infile = open(FONTS_PATH + 'multifont_mapping.pkl', 'rb')
  unicode_mapping_dict = pickle.load(infile)
  infile.close()
  for i in unicode_mapping_dict.keys():
    try:
      a = random.randint(0, len(unicode_mapping_dict[i]) - 1)
      # print(a)
      transformImg(drawChar(img_size, chr(i), font_size, FONTS_PATH + unicode_mapping_dict[i][a]))
    except (ValueError, OSError) as e:
      print(e, i, len(unicode_mapping_dict[i]))


def test_try_drawing(font_size=.2, img_size=200):
  # 17 corrupt unicode chars
  # 3600 invalid pixel size
  infile = open(FONTS_PATH + 'multifont_mapping.pkl', 'rb')
  unicode_mapping_dict = pickle.load(infile)
  infile.close()
  empty_image = np.full([img_size, img_size], 255)
  pop = list(unicode_mapping_dict.keys())
  random.shuffle(pop)
  for i in pop:
    try:
      # print(a)
      img = try_draw_char(i, unicode_mapping_dict[i], empty_image, img_size, font_size)
      cv.imshow('img', img)
      cv.waitKey(0)
    except (ValueError, OSError) as e:
      print(e, i, len(unicode_mapping_dict[i]))


def test_try_drawing_matplotlib(font_size=.2, img_size=200):
  # 17 corrupt unicode chars
  # 3600 invalid pixel size
  infile = open(FONTS_PATH + 'multifont_mapping.pkl', 'rb')
  unicode_mapping_dict = pickle.load(infile)
  infile.close()
  empty_image = np.full([img_size, img_size], 255, dtype=np.uint8)
  plt.imshow(empty_image)
  plt.show()
  pop = list(unicode_mapping_dict.keys())
  random.shuffle(pop)
  for i in pop:
    try:
      # print(a)
      img = try_draw_char(i, unicode_mapping_dict[i], empty_image, img_size, font_size)
      plt.imshow(img)
      plt.show()
    except (ValueError, OSError) as e:
      print(e, i, len(unicode_mapping_dict[i]))


def display_chars(display_train, display_test, font_size=.2, img_size=200):
  anchors, positives, negatives, x1_test, x2_test, y_test = compile_datasets(display_train, display_test, font_size,
                                                                             img_size, color_format='RGB')
  for i in range(display_train):
    print(anchors[i].dtype)
    cv.imshow('anchor', anchors[i])
    cv.imshow('positive', positives[i])
    cv.imshow('negative', negatives[i])
    cv.waitKey(0)
    cv.destroyAllWindows()
  for i in range(display_test):
    cv.imshow('x1', x1_test[i])
    cv.imshow('x2', x2_test[i])
    cv.waitKey(0)
    cv.destroyAllWindows()


def display_triplets_data_sample():
  preprocess_triplets = lambda x, y, z: (x, y, z)
  img_size = 100
  font_size = 0.4
  batch_size = 2
  triplets_dataset = get_triplet_tf_dataset(img_size, font_size, preprocess_fn=preprocess_triplets,
                                            batch_size=batch_size)
  for d in triplets_dataset:
    for b in range(batch_size):
      fig = plt.figure()
      fig.add_subplot(1, 3, 1)
      plt.imshow(d[0][b].numpy())
      fig.add_subplot(1, 3, 2)
      plt.imshow(d[1][b].numpy())
      fig.add_subplot(1, 3, 3)
      plt.imshow(d[2][b].numpy())
      plt.show()


def display_pairs_data_sample():
  preprocess_triplets = lambda x, y, z: (x, y, z)
  img_size = 100
  font_size = 0.4
  batch_size = 2
  triplets_dataset = get_balanced_pair_tf_dataset(img_size, font_size, preprocess_fn=preprocess_triplets,
                                                  batch_size=batch_size)
  for d in triplets_dataset:
    for b in range(batch_size):
      cv.imshow('x1', d[0][b].numpy())
      cv.imshow('x2', d[1][b].numpy())
      cv.waitKey(0)
      cv.destroyAllWindows()


if __name__ == '__main__':
  # Tests drawing each unicode character with a random font
  # test_drawing(.4,200)
  # test_try_drawing()
  # With OpenCV, display 10 training triplets and 5 testing pairs
  # display_chars(10, 10, .6, 200)

  # test Dataset
  display_triplets_data_sample()
  # display_pairs_data_sample()
  # ds = get_balanced_pair_tf_dataset(100, 0.4)
  # i = 0
  # for datapoint in ds:
  #   print(datapoint)
  #   i += 1
  #   if i == 3:
  #     break
