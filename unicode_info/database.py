"""
file: database.py
language:python 3

extracts information from the unicode consortium web database
"""

__DB = "https://www.unicode.org/Public/UCD/latest/ucd/"

from typing import *
import urllib.request
import requests
import pickle
import numpy as np
import random

# inclusive decimal range of a unicode subset
__UnicodeRange = Tuple[int, int]
# matches subsets names with codepoint ranges
__UnicodeBlocks = Dict[str, __UnicodeRange]
# maps potentially implemented characters to unicode
# blocks with implemented characters
__UnicodeMap = List[str]

UNDEFINED_BLOCK = "undefined"  # for indicating that a character is not defined


def generate_positive_pairs_consortium(unicode_clusters_codepoints_map: dict, num_pairs: int):
  """
  randomly sample positive pairs of homoglyphs with replacement, returns list of tuples of codepoint_id, codepoint_id

  clusters_codepoints_map: mapping of cluster ids to lists of codepoints
  """
  reverse_mapping = {codepoint: cluster_id for cluster_id in unicode_clusters_codepoints_map.keys() for codepoint in
                     unicode_clusters_codepoints_map[cluster_id]}
  codepoints = list(reverse_mapping.keys())
  l = [None] * num_pairs
  for i in range(num_pairs):
    code_a = codepoints[random.randint(0, len(codepoints) - 1)]
    code_b = code_a
    homoglyphs = unicode_clusters_codepoints_map[reverse_mapping[code_a]]
    while code_b == code_a:
      code_b = homoglyphs[random.randint(0, len(homoglyphs) - 1)]
    l[i] = (code_a, code_b)
  return l


def generate_negative_pairs_consortium(unicode_clusters_codepoints_map: dict, num_pairs: int):
  """
  randomly sample positive pairs of homoglyphs with replacement, returns list of tuples of codepoint_id, codepoint_id

  clusters_codepoints_map: mapping of cluster ids to lists of codepoints
  """
  reverse_mapping = {codepoint: cluster_id for cluster_id in unicode_clusters_codepoints_map.keys() for codepoint in
                     unicode_clusters_codepoints_map[cluster_id]}
  codepoints = list(reverse_mapping.keys())
  l = [None] * num_pairs
  for i in range(num_pairs):
    code_a = codepoints[random.randint(0, len(codepoints) - 1)]
    code_b = code_a
    while reverse_mapping[code_b] == reverse_mapping[code_a]:
      code_b = codepoints[random.randint(0, len(codepoints) - 1)]
    l[i] = (code_a, code_b)
  return l


def get_consortium_clusters_dict():
  url = 'https://www.unicode.org/Public/security/12.0.0/confusables.txt'
  # Load Text File
  r = requests.get(url, allow_redirects=True)
  punycodes_list = r.text[385:].split("\n")
  punycodes_list = punycodes_list[:len(punycodes_list) - 3]
  # 4248 pairs
  # 3353 we support :(
  consortium_clusters_dict = {}
  for i in punycodes_list:
    if i is not '':
      punycode_pair = i.split(";")[:2]
      if len(punycode_pair[1]) < 8:
        source = str(int(punycode_pair[0].replace(' ', ''), 16))
        target = str(int(punycode_pair[1].replace(' ', '').replace('\t', ''), 16))
        if target not in consortium_clusters_dict:
          consortium_clusters_dict[target] = [source]
        else:
          consortium_clusters_dict[target].append(source)
  return consortium_clusters_dict


def generate_supported_consortium_feature_vectors_and_clusters_dict(n_clusters: int, features_dict_file_path: str):
  consortium_clusters_dict = get_consortium_clusters_dict()
  # print(len(consortium_clusters_dict))
  features_dict = pickle.load(open(features_dict_file_path, 'rb'))
  supported_consortium_feature_vectors = {}
  supported_consortium_clusters_dict = {}
  for cluster_source in consortium_clusters_dict.keys():
    if cluster_source in features_dict:
      supported_consortium_clusters_dict[cluster_source] = []
      for target in consortium_clusters_dict[cluster_source]:
        if target in features_dict:
          supported_consortium_clusters_dict[cluster_source].append(target)
          if cluster_source not in supported_consortium_feature_vectors:
            supported_consortium_feature_vectors[cluster_source] = features_dict[cluster_source]
          if target not in supported_consortium_feature_vectors:
            supported_consortium_feature_vectors[target] = features_dict[target]
      if len(supported_consortium_clusters_dict[cluster_source]) == 0:
        del supported_consortium_clusters_dict[cluster_source]
    if len(supported_consortium_clusters_dict) == n_clusters:
      break
  reformatted = {}
  count = 0
  for key, value in supported_consortium_clusters_dict.items():
    value.append(key)
    reformatted[count] = value
    count += 1
  supported_consortium_clusters_dict = reformatted
  return supported_consortium_feature_vectors, supported_consortium_clusters_dict


def generate_data_for_experiment(num_random_additions: int = 0):
  supported_consortium_feature_vectors, supported_consortium_clusters_dict = generate_supported_consortium_feature_vectors_and_clusters_dict(
    9999, 'features_dict_file.pkl')
  clusters_unicode_characters = list(supported_consortium_feature_vectors.keys())
  unicode_median_feature_vector_dict = pickle.load(open('features_dict_file.pkl', 'rb'))
  all_unicode_characters = list(unicode_median_feature_vector_dict.keys())
  for _ in range(num_random_additions):
    possible_random_character = all_unicode_characters.pop(random.randint(0, len(all_unicode_characters) - 1))
    if possible_random_character not in clusters_unicode_characters:
      supported_consortium_feature_vectors[possible_random_character] = unicode_median_feature_vector_dict[
        possible_random_character]
  return supported_consortium_feature_vectors, supported_consortium_clusters_dict


def _generate_statistics(converted_dict, features_dict_file_path):
  import cupy as cp
  def _generate_adjacency_matrix(features):
    # gpu [n, k]
    ordered_features_gpu = cp.array(features)
    n, k = ordered_features_gpu.shape

    a = ordered_features_gpu.reshape((n, 1, 1, k))
    b = ordered_features_gpu.reshape((1, n, k, 1))
    # [n, n]
    dot_products = cp.matmul(a, b).reshape((n, n))

    # [n]
    norms = cp.linalg.norm(ordered_features_gpu, axis=1)

    norms_a = norms.reshape((n, 1))
    norms_b = norms.reshape((1, n))  # same as the above but transposed
    # [n, n]
    norms_prod = cp.multiply(norms_a, norms_b)
    cosine_similarity = dot_products / (norms_prod + 1e-7)
    return cp.asnumpy(cosine_similarity)
  features_dict = pickle.load(open(features_dict_file_path, 'rb'))
  mean = 0
  count = 0
  std_dev = 0
  for cluster in converted_dict.values():
    ordered_features = np.empty([len(cluster), len(features_dict[cluster[0]])], dtype=np.float32)
    for i in range(len(cluster)):
      ordered_features[i] = features_dict[cluster[i]]
    if len(cluster) > 2:
      cos_matrix = _generate_adjacency_matrix(ordered_features)
      stats = np.tril(cos_matrix, -1)
      stats = stats[stats != 0]
      mean += np.mean(stats)
      std_dev += np.std(stats)
      count += 1
  print(mean / count)
  print(std_dev / count)


# def normalize_rows(vector):
#     return vector / (np.linalg.norm(vector, axis=1).reshape((vector.shape[0], 1)) + 1e-8)
#
#
# def calculate_centroid(feature_vectors):
#     normalized_vectors = normalize_rows(feature_vectors)
#     centroid_ = np.sum(normalized_vectors, axis=0)
#     return normalize_rows(centroid_.reshape((1, -1)))


def _is_character_block(block_name: str) -> bool:
  """
  checks if block implements actual characters
  :param block_name: name of the block
  :return: true if are characters
  """
  keywords = ["Surrogate", "Private"]
  is_character = True
  for keyword in keywords:
    if keyword in block_name:
      is_character = False
      break
  return is_character


def map_blocks() -> (__UnicodeBlocks, __UnicodeMap, int):
  """
  In some browsers and email clients, all the characters in the domain name
  needs to be of the same subset, or "block," of unicode, in order to be
  displayed in unicode form.

  This helper function determines whether a unicode domain name will be
  displayed in "xe--*" ascii encoding format, or its unicode form.

  This function uses the definition of subset blocks specified by the latest
  unicode standard.

  :return: tuple of UnicodeSets, UnicodeMap, and total # of chars
  """
  blocks = {}
  block_map = []
  with urllib.request.urlopen(__DB + "Blocks.txt") as response:
    lines = response.read().decode('utf-8').split("\n")
    for line in lines:
      if len(line) > 0 and line[0] != '\n' and line[0] != '#':
        line = line.strip()
        (hex_range, block_name) = line.split("; ")
        if _is_character_block(block_name):
          (start_hex, end_hex) = hex_range.split("..")
          start = int(start_hex, 16)
          end = int(end_hex, 16)
          blocks[block_name] = (start, end)
          if len(block_map) < end + 1:
            for i in range(len(block_map), end + 1):
              block_map.append(UNDEFINED_BLOCK)
          for i in range(start, end + 1):
            block_map[i] = block_name
  # as of unicode 12
  # block_map produces an array for the first 900k unicode code points
  # around 140k of which belong to blocks with defined code points
  n = _prune_block_map(block_map)
  return blocks, block_map, n


def _is_code_range(description: str) -> int:
  """
  determines whether an entry is a code point or the start/end of a range
  :param description: entry description, second field in line
  :return: -1 if it's a code point, 0 if it's first in a range, 1 if it's last
  """
  if len(description) > 4:
    if description[-4:] == "rst>":  # first in range, inclusive
      return 0
    if description[-4:] == "ast>":  # last in range, inclusive
      return 1
  return -1


def _prune_block_map(block_map: __UnicodeMap):
  """
  goes through the block map and "un-define" the blocks for characters
  that are not actually implemented
  :param block_map: unicode map of characters and blocks
  :return: total number of implemented characters
  """
  n = 0
  implemented = [False] * len(block_map)
  with urllib.request.urlopen(__DB + "UnicodeData.txt") as response:
    lines = response.read().decode('utf-8').split("\n")
    i = 0
    while i < len(lines):
      line = lines[i].strip()
      fields = line.split(";")
      if len(line) > 0 and fields[1] != "<control>" \
          and _is_character_block(fields[1]):
        index = int(fields[0], 16)
        retval = _is_code_range(fields[1])
        if retval == -1:
          implemented[index] = True
          n += 1
        elif retval == 0:
          i += 1
          line = lines[i].strip()
          fields = line.split(";")
          end = int(fields[0], 16)
          for k in range(index, end + 1):
            implemented[k] = True
            n += 1
      i += 1
  for i in range(len(implemented)):
    if not implemented[i]:
      block_map[i] = UNDEFINED_BLOCK
  # 137929 as of unicode 12
  return n
