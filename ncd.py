# see first article in https://www.springerprofessional.de/en/web-information-system-engineering-wise-2011/3755202?tocPage=1
# for context, ncd stands for normalized compression distance

import lzma
import numpy as np
import random

from sklearn import svm
from sklearn.preprocessing import normalize
from sklearn.metrics import average_precision_score, precision_recall_curve, plot_precision_recall_curve
from matplotlib import pyplot as plt

from unicode_cons import generate_supported_consortium_feature_vectors_and_clusters_dict, convert
from generate_datasets import try_draw_single_font


def C(x:bytes):
  """
  gives the compressed length of a byte string
  """
  return len(lzma.compress(x))


def ncd(x:bytes, y:bytes):
  """
  calculates the normalized compression distance between two bytes string
  """
  Cx = C(x)
  Cy = C(y)
  Cxy = C(x + y)
  return (Cxy - min(Cx, Cy)) / max(Cx, Cy)


def ncd_ndarray(x:np.ndarray, y:np.ndarray):
  return ncd(x.tobytes(), y.tobytes())


def generate_positive_pairs_consortium(unicode_clusters_codepoints_map:dict, num_pairs:int):
  """
  randomly sample positive pairs of homoglyphs with replacement, returns list of tuples of codepoint_id, codepoint_id

  clusters_codepoints_map: mapping of cluster ids to lists of codepoints
  """
  reverse_mapping = {codepoint:cluster for cluster in unicode_clusters_codepoints_map.values() for codepoint in cluster}
  codepoints = list(reverse_mapping.keys())
  l = [None] * num_pairs
  for i in range(num_pairs):
    code_a = codepoints[random.randint(0, len(codepoints) - 1)]
    code_b = code_a
    while code_b == code_a:
      homoglyphs = unicode_clusters_codepoints_map[reverse_mapping[code_a]]
      code_b = homoglyphs[random.randint(0, len(homoglyphs) - 1)]
    l[i] = (code_a, code_b)
  return l


def generate_negative_pairs_consortium(unicode_clusters_codepoints_map:dict, num_pairs:int):
  """
  randomly sample positive pairs of homoglyphs with replacement, returns list of tuples of codepoint_id, codepoint_id

  clusters_codepoints_map: mapping of cluster ids to lists of codepoints
  """
  reverse_mapping = {codepoint:cluster for cluster in unicode_clusters_codepoints_map.values() for codepoint in cluster}
  codepoints = list(reverse_mapping.keys())
  l = [None] * num_pairs
  for i in range(num_pairs):
    code_a = codepoints[random.randint(0, len(codepoints) - 1)]
    code_b = code_a
    while reverse_mapping[code_b] == reverse_mapping[code_a]:
      code_b = codepoints[random.randint(0, len(codepoints) - 1)]
    l[i] = (code_a, code_b)
  return l


def train_svm_generate_statistics_and_auc(measures:np.ndarray, labels:np.ndarray):
  """
  dim measures = [n], dtype = float
  dim labels = [n], dtype = int
  """
  classifier = svm.SVC(kernel='linear')
  measures = normalize(measures)
  y_score = classifier.fit(measures, labels).decision_function(labels)

  average_precision = average_precision_score(labels, y_score)
  disp = plot_precision_recall_curve(classifier, labels, y_score)
  print('Average precision-recall score: {0:0.2f}'.format(
    average_precision))
  plt.show()


def comparison():
  num_pairs = 100

  supported_consortium_feature_vectors, supported_consortium_clusters_dict = generate_supported_consortium_feature_vectors_and_clusters_dict(9999, 'features_dict_file.pkl')
  ground_truth_consortium_codepoints_map = convert(supported_consortium_clusters_dict)

  min_supported_fonts_dict = pickle.load(open('min_supported_fonts.pkl', 'rb'))
  
  # Load model input info
  model_info_file = open(os.path.join('model_1', 'model_info.pkl'), 'rb')
  model_info_dict = pickle.load(model_info_file)
  img_size, font_size = model_info_dict['img_size'], model_info_dict['font_size']
  empty_image = np.full((img_size, img_size), 255)

  positive_pairs = generate_negative_pairs_consortium(ground_truth_consortium_codepoints_map, num_pairs)
  negative_pairs = generate_negative_pairs_consortium(ground_truth_consortium_codepoints_map, num_pairs)
  pairs = positive_pairs + negative_pairs

  labels = np.zeros(2000, dtype=int)
  labels[0:1000] = 1

  cosine_similarities = np.empty(num_pairs * 2, dtype=float)
  normalized_compression_distances = np.empty(num_pairs * 2, dtype=float)

  for i in range(num_pairs * 2):
    code_x, code_y = pairs[i]
    features_x, features_y = supported_consortium_feature_vectors[code_x], supported_consortium_feature_vectors[code_y]
    glyph_x, glyph_y = try_draw_single_font(code_x, min_supported_fonts_dict[code_x], empty_image, img_size, "./fonts", transform_img=False), try_draw_single_font(code_y, min_supported_fonts_dict[code_y], empty_image, img_size, "./fonts", transform_img=False)
    cosine_similarities[i] = np.dot(features_x, features_y) / (np.linalg.norm(features_x) * np.linalg.norm(features_y))
    normalized_compression_distances[i] = ncd_ndarray(glyph_x, glyph_y)

  print("DEEP")
  train_svm_generate_statistics_and_auc(cosine_similarities, labels)
  print("NCD")
  train_svm_generate_statistics_and_auc(normalized_compression_distances, labels)
