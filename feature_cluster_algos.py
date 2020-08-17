import pickle
import os
import numpy as np
import tensorflow as tf
import cv2 as cv

from generate_datasets import try_draw_single_font
from unicode_info.database import generate_data_for_experiment
from cluster_metrics import calculate_mean_iou, calculate_mean_precision

import time
import datetime

import argparse

def generate_features_dict_file_path(save_dir: str, features_dict_file="features_dict_file.pkl"):
  return os.path.join(save_dir, features_dict_file)


def generate_codepoints_cluster_map_file_path(save_dir: str): return os.path.join(save_dir,
                                                                                  "codepoints_cluster_map.pkl")


def generate_cluster_codepoints_map_file_path(save_dir: str): return os.path.join(save_dir,
                                                                                  "codepoints_cluster_map.pkl")


class EfficientNetFeatureExtractor:
  def __init__(self, model_path: str, batch_size: int, save_dir: str, multifont_mapping_path: str):
    self.mp = model_path
    self.bs = batch_size
    self.sd = save_dir
    self.mmp = multifont_mapping_path

  # Returns: A dictionary - Keys are all supported unicode characters, Values are minimum supported font for that unicode
  def generate_minimum_used_fonts_dict(self, unicode_supported_fonts_drawn_dict) -> dict:
    num_unicodes = len(unicode_supported_fonts_drawn_dict)
    minimum_font_dict = {}
    for unicode, font_feature_dict in unicode_supported_fonts_drawn_dict.items():
      supported_fonts = font_feature_dict.keys()
      for font in supported_fonts:
        if font not in minimum_font_dict:
          minimum_font_dict[font] = [unicode]
        else:
          minimum_font_dict[font].append(unicode)
    font_unicodes_list = list(minimum_font_dict.items())
    minimum_used_fonts_dict = {}
    last_update = []
    copied_font_unicodes_list = font_unicodes_list.copy()
    while len(minimum_used_fonts_dict) < num_unicodes:
      max_len_index = -1
      max_len = 0
      for char_ in range(len(copied_font_unicodes_list)):
        copied_font_unicodes_list[char_] = (copied_font_unicodes_list[char_][0],
                                            [i for i in copied_font_unicodes_list[char_][1] if
                                             i not in last_update])
        font_len = len(copied_font_unicodes_list[char_][1])
        if font_len > max_len:
          max_len = font_len
          max_len_index = char_
      most_supported_font = copied_font_unicodes_list[max_len_index]
      for unicode in most_supported_font[1]:
        assert unicode not in minimum_used_fonts_dict
        minimum_used_fonts_dict[unicode] = most_supported_font[0]
      last_update = most_supported_font[1]
      del copied_font_unicodes_list[max_len_index]
    return minimum_used_fonts_dict

  def _load_model_load_data_and_extract_features(self, model_path: str, batch_size: int,
                                                 multifont_mapping_path: str) -> dict:
    # Load model + weights
    json_file = open(os.path.join(model_path, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(model_path, 'model.h5'))

    # Load model input info
    model_info_file = open(os.path.join(model_path, 'model_info.pkl'), 'rb')
    model_info_dict = pickle.load(model_info_file)
    img_size, font_size = model_info_dict['img_size'], model_info_dict['font_size']
    empty_image = np.full((img_size, img_size), 255)

    # Load multifont mapping dict
    multifont_mapping_file = open(multifont_mapping_path, 'rb')
    unicode_font_mapping_dict = pickle.load(multifont_mapping_file)

    if os.path.exists('unicode_supported_fonts_drawn_dict.pkl'):
      unicode_supported_fonts_drawn_dict_file = open('unicode_supported_fonts_drawn_dict.pkl', 'rb')
      unicode_supported_fonts_drawn_dict = pickle.load(unicode_supported_fonts_drawn_dict_file)
    else:
      unicode_supported_fonts_drawn_dict = {}
      for unicode_point in unicode_font_mapping_dict.keys():
        print(unicode_point)
        for font_path in unicode_font_mapping_dict[unicode_point]:
          unicode_drawn = try_draw_single_font(unicode_point, font_path, empty_image, img_size, font_size,
                                               "./fonts",
                                               transform_img=False)
          if not (unicode_drawn == empty_image).all():
            unicode_drawn_preprocessed = (
                (cv.cvtColor(unicode_drawn, cv.COLOR_GRAY2RGB) - (255 / 2)) / (255 / 2)).astype(np.float32)
            if unicode_point not in unicode_supported_fonts_drawn_dict:
              unicode_supported_fonts_drawn_dict[unicode_point] = {font_path: unicode_drawn_preprocessed}
            else:
              unicode_supported_fonts_drawn_dict[unicode_point][font_path] = unicode_drawn_preprocessed
      with open('unicode_supported_fonts_drawn_dict.pkl', 'wb+') as f:
        pickle.dump(unicode_supported_fonts_drawn_dict, f)
    print(len(unicode_supported_fonts_drawn_dict))

    if not os.path.exists('unicode_minimum_font_dict.pkl'):
      minimum_used_fonts_dict = self.generate_minimum_used_fonts_dict(unicode_supported_fonts_drawn_dict)
      with open('unicode_minimum_font_dict.pkl', 'wb+') as f:
        pickle.dump(minimum_used_fonts_dict, f)
    else:
      minimum_used_fonts_dict_file = open('unicode_minimum_font_dict.pkl', 'rb')
      minimum_used_fonts_dict = pickle.load(minimum_used_fonts_dict_file)
    print(len(minimum_used_fonts_dict))

    unicode_batch = {}
    unicode_feature_vectors_dict = {}
    for unicode_point in unicode_supported_fonts_drawn_dict.keys():
      unicode_batch[unicode_point] = unicode_supported_fonts_drawn_dict[unicode_point][
        minimum_used_fonts_dict[unicode_point]]
      if len(unicode_batch) == batch_size:
        unicode_batch_forward = loaded_model.predict(tf.convert_to_tensor(list(unicode_batch.values())))
        unicode_batch = dict(zip(unicode_batch.keys(), unicode_batch_forward))
        unicode_feature_vectors_dict.update(unicode_batch)
        unicode_batch = {}
    if len(unicode_batch) > 0:
      unicode_batch_forward = loaded_model.predict(tf.convert_to_tensor(list(unicode_batch.values())))
      unicode_batch = dict(zip(unicode_batch.keys(), unicode_batch_forward))
      unicode_feature_vectors_dict.update(unicode_batch)
    print(len(unicode_feature_vectors_dict))
    return unicode_feature_vectors_dict

  def extract_and_save_features(self):
    features_dict = self._load_model_load_data_and_extract_features(self.mp, self.bs, self.mmp)
    with open(generate_features_dict_file_path(self.sd), 'wb+') as f:
      pickle.dump(features_dict, f)


def cosine_similarity_matrix_cpu(features: np.ndarray) -> np.ndarray:
  start_time = time.time()
  n, k = features.shape

  a_ = features.reshape((n, 1, 1, k))
  b_ = features.reshape((1, n, k, 1))
  # [n, n]
  dot_products = np.matmul(a_, b_).reshape((n, n))

  # [n]
  norms = np.linalg.norm(features, axis=1)

  norms_a = norms.reshape((n, 1))
  norms_b = norms.reshape((1, n))
  # [n, n]
  norms_prod = np.multiply(norms_a, norms_b)
  cosine_similarity = dot_products / norms_prod

  elapsed_seconds = time.time() - start_time
  print("Time spent on similarity matrix: " + str(datetime.timedelta(seconds=elapsed_seconds)))
  return cosine_similarity


class _AbstractFeatureClusterer:
  def __init__(self, save_dir: str):
    self.sd = save_dir
    with open(generate_features_dict_file_path(self.sd), 'rb') as f:
      self.features_dict = pickle.load(f)

  def cluster_features_into_equivalence_classes(self, features_dict: dict) -> (dict, dict):
    """
    *
    :param features_dict: keys are codepoint integers, values are numpy arrays of identical dimension
    :return: mapping of codepoints to cluster id, mapping of cluster ids to lists of codepoints
    """
    raise NotImplementedError

  def find_and_save_equivalence_classes(self):
    codepoints_cluster_map, cluster_codepoints_map = self.cluster_features_into_equivalence_classes(
      self.features_dict)
    with open(generate_codepoints_cluster_map_file_path(self.sd), 'wb+') as f:
      pickle.dump(codepoints_cluster_map, f)
    with open(generate_cluster_codepoints_map_file_path(self.sd), 'wb+') as f:
      pickle.dump(cluster_codepoints_map, f)


def _dfs_traverse(adj_mat: np.ndarray, visited: np.ndarray, node: int, trace: list):
  visited[node] = True
  trace.append(node)
  neighbors_indices = np.nonzero(adj_mat[node])[0]
  for i in neighbors_indices:
    if not visited[i]:
      _dfs_traverse(adj_mat, visited, i, trace)


def _find_nontrivial_components_from_adjacency_matrix(adj_mat: np.ndarray) -> list:
  """
  a simple algorithm to iterate through all unvisited nodes and DFS through their neighbors to find components
  ignores components with only one node
  :param adj_mat: *
  :return: list of lists of node ids, corresponding to connected components
  """
  n, _ = adj_mat.shape
  visited = np.zeros(n, dtype=np.bool)
  connected_components = []
  for node in range(n):
    if not visited[node]:
      trace = []
      _dfs_traverse(adj_mat, visited, node, trace)
      if len(trace) > 0:
        connected_components.append(trace)
  return connected_components


class _AbstractGraphClusterer(_AbstractFeatureClusterer):
  """
  cluster unicodes
  1. compute adjacency matrix from pairwise comparison
  2. find connected components with more than one nodes
  """

  def __init__(self, save_dir: str):
    super().__init__(save_dir)

  def _generate_adjacency_matrix(self, features: np.ndarray):
    raise NotImplementedError

  def cluster_features_into_equivalence_classes(self, features_dict: dict) -> (dict, dict):
    """
    *
    :param features_dict: keys are codepoint integers, values are numpy arrays of identical dimension
    :return: mapping of codepoints to cluster id, a mapping of cluster ids to lists of codepoints
    """
    # cpu [n]
    ordered_codepoints = list(features_dict.keys())
    ordered_features = np.stack(list(features_dict.values()))
    adj_mat = self._generate_adjacency_matrix(ordered_features)
    connected_components = _find_nontrivial_components_from_adjacency_matrix(adj_mat)
    codepoints_cluster_map = {}
    cluster_codepoints_map = {}
    for cluster_id, component in enumerate(connected_components):
      cluster_codepoints_map[cluster_id] = []
      for node in component:
        codepoint = ordered_codepoints[node]
        codepoints_cluster_map[codepoint] = cluster_id
        cluster_codepoints_map[cluster_id].append(codepoint)
    for cluster_id_neg, codepoint in enumerate(ordered_codepoints):
      if codepoint not in codepoints_cluster_map:
        cluster_id = -1 * cluster_id_neg
        codepoints_cluster_map[codepoint] = cluster_id
        cluster_codepoints_map[cluster_id] = [codepoint]
    return codepoints_cluster_map, cluster_codepoints_map


class CosineSimGraphClustererCPU(_AbstractGraphClusterer):
  def __init__(self, save_dir: str, threshold: float, epsilon: float):
    super().__init__(save_dir)
    self.threshold = threshold
    self.epsilon = epsilon

  def _generate_adjacency_matrix(self, features: np.ndarray):
    self.cosine_similarity = cosine_similarity_matrix_cpu(features)
    adjacency_matrix = (self.cosine_similarity > self.threshold)
    return adjacency_matrix


def greedy_clique_cluster_heuristic(features_dict: dict, target_mean: float, target_std: float,
                                    target_threshold: float):
  """
  Ground Truth Mean: 0.767601996660232
  Ground Truth Std: 0.0935437240793059
  Optimized parameters: .72,.01,.94
  """
  ordered_codepoints = list(features_dict.keys())
  ordered_features = np.stack(list(features_dict.values()))
  similarity_matrix = cosine_similarity_matrix_cpu(ordered_features)

  start_time = time.time()
  cluster_id_indices_map = {}
  for node_index in range(len(ordered_codepoints)):
    added_to_existing_cluster = False
    for cluster_id, cluster_indices in cluster_id_indices_map.items():
      edges = similarity_matrix[node_index, cluster_indices]
      if (edges > target_threshold).all():
        cluster_id_indices_map[cluster_id].append(node_index)
        added_to_existing_cluster = True
        break
    if not added_to_existing_cluster:
      cluster_id_indices_map[len(cluster_id_indices_map)] = [node_index]
  print("Time spent on finding cliques: " + str(datetime.timedelta(seconds=time.time() - start_time)))

  start_time = time.time()
  id_upperbound = len(cluster_id_indices_map)
  for cluster_id in range(id_upperbound):
    if cluster_id not in cluster_id_indices_map: continue
    cluster_indices = cluster_id_indices_map[cluster_id]
    for merge_candidate_id in range(cluster_id + 1, id_upperbound):
      if merge_candidate_id not in cluster_id_indices_map: continue
      candidate_indices = cluster_id_indices_map[merge_candidate_id]
      # https://stackoverflow.com/questions/22927181/selecting-specific-rows-and-columns-from-numpy-array
      edges = similarity_matrix[np.ix_(cluster_indices, candidate_indices)]
      edges_mean = np.mean(edges)
      edges_std = np.std(edges)
      if edges_mean > target_mean and edges_std < target_std:
        cluster_id_indices_map[cluster_id].extend(candidate_indices)
        del cluster_id_indices_map[merge_candidate_id]
  print("Time spent on merging clusters: " + str(datetime.timedelta(seconds=time.time() - start_time)))
  predicted_cluster_codepoints_map = {cluster_id: [ordered_codepoints[index] for index in indices]
                                      for cluster_id, indices in cluster_id_indices_map.items()}
  predicted_codepoints_cluster_map = {ordered_codepoints[index]: cluster_id
                                      for cluster_id, indices in cluster_id_indices_map.items()
                                      for index in indices}
  return predicted_codepoints_cluster_map, predicted_cluster_codepoints_map


def _test_dfs_components_finder():
  import matplotlib.pyplot as plt
  import networkx as nx

  def show_graph_with_labels(adjacency_matrix):
    n, _ = adjacency_matrix.shape
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw_networkx(gr, node_size=250, labels={i: str(i) for i in range(n)}, with_labels=True)
    plt.show()

  adj_mat = np.zeros(shape=[10, 10], dtype=np.bool)
  for i in range(10):
    adj_mat[i, i] = True
  # component 1: 1, 2, 3
  adj_mat[1, 3] = True
  adj_mat[3, 1] = True
  adj_mat[1, 2] = True
  adj_mat[2, 1] = True
  adj_mat[2, 3] = True
  adj_mat[3, 2] = True
  # component 2: 4, 5
  adj_mat[4, 5] = True
  adj_mat[5, 4] = True
  # component 3: 6, 7, 8, 9
  adj_mat[6, 7] = True
  adj_mat[7, 6] = True
  adj_mat[7, 8] = True
  adj_mat[8, 7] = True
  adj_mat[8, 9] = True
  adj_mat[9, 8] = True
  print(_find_nontrivial_components_from_adjacency_matrix(adj_mat))
  show_graph_with_labels(adj_mat)


def run_dfs_on_consortium(num_random_additions: int = 0):
  supported_consortium_feature_vectors, ground_truth_consoritium_codepoints_map = generate_data_for_experiment(
    num_random_additions)
  Cluster_Algo = CosineSimGraphClustererCPU(save_dir="./", threshold=.92, epsilon=1e-5)
  predicted_codepoints_cluster_map, predicted_cluster_codepoints_map = Cluster_Algo.cluster_features_into_equivalence_classes(
    features_dict=supported_consortium_feature_vectors)
  mean_IOU, mean_precision = calculate_mean_iou(predicted_codepoints_cluster_map,
                                                predicted_cluster_codepoints_map,
                                                ground_truth_consoritium_codepoints_map), calculate_mean_precision(
    predicted_codepoints_cluster_map,
    predicted_cluster_codepoints_map,
    ground_truth_consoritium_codepoints_map)
  print(f"Mean IOU: " + str(mean_IOU))
  print(f"Mean precision: " + str(mean_precision))


def run_clique_on_consortium(num_random_additions: int = 0):
  supported_consortium_feature_vectors, ground_truth_consoritium_codepoints_map = generate_data_for_experiment(
    num_random_additions)
  predicted_codepoints_cluster_map, predicted_cluster_codepoints_map = \
    greedy_clique_cluster_heuristic(supported_consortium_feature_vectors, 0.72, 0.01, 0.94)
  mean_IOU, mean_precision = calculate_mean_iou(predicted_codepoints_cluster_map,
                                                predicted_cluster_codepoints_map,
                                                ground_truth_consoritium_codepoints_map), calculate_mean_precision(
    predicted_codepoints_cluster_map,
    predicted_cluster_codepoints_map,
    ground_truth_consoritium_codepoints_map)
  print(f"Mean IOU: " + str(mean_IOU))
  print(f"Mean precision: " + str(mean_precision))


if __name__ == "__main__":
  # _test_dfs_components_finder()
  parser = argparse.ArgumentParser()
  parser.add_argument('-hc', '--heuristic_choice', action='store', type=str, default='dfs')
  parser.add_argument('-nra', '--num_rand_add', action='store', type=int, default=0)
  args = parser.parse_args()
  num_rand_add = args.num_rand_add
  hc = args.heuristic_choice

  t0 = time.time()
  print("Heuristic: " + hc)
  print("Random Additions: " + str(num_rand_add))

  if hc == "clique":
    run_clique_on_consortium(num_rand_add)
  elif hc == "dfs":
    run_dfs_on_consortium(num_rand_add)
  print("Total Elapsed Time: " + str(datetime.timedelta(seconds=time.time() - t0)))
