import pickle
import os
import numpy as np
import cupy as cp


def generate_features_dict_file_path(save_dir:str): return os.path.join(save_dir, "codepoints_features_map.pkl")


def generate_codepoints_cluster_map_file_path(save_dir:str): return os.path.join(save_dir, "codepoints_cluster_map.pkl")


def generate_cluster_codepoints_map_file_path(save_dir:str): return os.path.join(save_dir, "codepoints_cluster_map.pkl")


# TODO: implement this for efficient net
class _AbstractFeatureExtractor:
  def __init__(self, model_path:str, batch_size:int, save_dir:str):
    self.mp = model_path
    self.bs = batch_size
    self.sd = save_dir

  def _load_model_load_data_and_extract_features(self, model_path:str, batch_size:int) -> dict:
    """
    *
    :param model_path: *
    :param batch_size: *
    :return: a dict, keys are codepoint integers, values are numpy arrays of identical dimension
    """
    raise NotImplementedError

  def extract_and_save_features(self):
    features_dict = self._load_model_load_data_and_extract_features(self.mp, self.bs)
    with open(generate_features_dict_file_path(self.sd), 'wb+') as f:
      pickle.dump(features_dict, f)


class _AbstractFeatureClusterer:
  def __init__(self, save_dir:str):
    self.sd = save_dir
    with open(generate_features_dict_file_path(self.sd), 'rb') as f:
      self.features_dict = pickle.load(f)

  def _cluster_features_into_equivalence_classes(self, features_dict:dict) -> (dict, dict):
    """
    *
    :param features_dict: keys are codepoint integers, values are numpy arrays of identical dimension
    :return: mapping of codepoints to cluster id, mapping of cluster ids to lists of codepoints
    """
    raise NotImplementedError

  def find_and_save_equivalence_classes(self):
    codepoints_cluster_map, cluster_codepoints_map = self._cluster_features_into_equivalence_classes(self.features_dict)
    with open(generate_codepoints_cluster_map_file_path(self.sd), 'wb+') as f:
      pickle.dump(codepoints_cluster_map, f)
    with open(generate_cluster_codepoints_map_file_path(self.sd), 'wb+') as f:
      pickle.dump(cluster_codepoints_map, f)


def _dfs_traverse(adj_mat:np.ndarray, visited:np.ndarray, node:int, trace:list):
  visited[node] = True
  trace.append(node)
  neighbors_indices = np.nonzero(adj_mat[node])[0]
  for i in neighbors_indices:
    if not visited[i]:
      _dfs_traverse(adj_mat, visited, i, trace)


def _find_nontrivial_components_from_adjacency_matrix(adj_mat:np.ndarray) -> list:
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
  def __init__(self, save_dir:str):
    super().__init__(save_dir)

  def _generate_adjacency_matrix(self, features:np.ndarray):
    raise NotImplementedError

  def _cluster_features_into_equivalence_classes(self, features_dict:dict) -> (dict, dict):
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
    return codepoints_cluster_map, cluster_codepoints_map


class CosineSimGraphClustererGPU(_AbstractGraphClusterer):
  def __init__(self, save_dir:str, threshold:float, epsilon:float):
    super().__init__(save_dir)
    self.threshold = threshold
    self.epsilon = epsilon

  def _generate_adjacency_matrix(self, features:np.ndarray):
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
    norms_b = norms.reshape((1, n)) # same as the above but transposed
    # [n, n]
    norms_prod = cp.multiply(norms_a, norms_b)
    cosine_similarity = dot_products / (norms_prod + self.epsilon)
    # cpu [n, n]
    adjacency_matrix = cp.asnumpy(cosine_similarity > self.threshold)
    return adjacency_matrix


class EuclidDistGraphClustererGPU(_AbstractGraphClusterer):
  def __init__(self, save_dir:str, threshold:float):
    super().__init__(save_dir)
    self.threshold = threshold

  def _generate_adjacency_matrix(self, features:np.ndarray):
    # gpu [n, k]
    ordered_features_gpu = cp.array(features)
    n, k = ordered_features_gpu.shape

    a = ordered_features_gpu.reshape((n, 1, k))
    b = ordered_features_gpu.reshape((1, n, k))
    # [n, n]
    l2_norm = cp.linalg.norm(a - b, axis=2)

    # cpu [n, n]
    adjacency_matrix = cp.asnumpy(l2_norm < self.threshold)
    return adjacency_matrix


def _test_dfs_components_finder():
  import matplotlib.pyplot as plt
  import networkx as nx

  def show_graph_with_labels(adjacency_matrix):
    n, _ = adjacency_matrix.shape
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw_networkx(gr, node_size=250, labels={i:str(i) for i in range(n)}, with_labels=True)
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


if __name__ == "__main__":
  _test_dfs_components_finder()
