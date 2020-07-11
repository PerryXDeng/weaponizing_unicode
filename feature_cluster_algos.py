import pickle
import os


def generate_features_dict_file_path(save_dir:str):
  return os.path.join(save_dir, "codepoints_features_map.pkl")


def generate_codepoints_cluster_map_file_path(save_dir:str):
  return os.path.join(save_dir, "codepoints_cluster_map.pkl")


def generate_cluster_codepoints_map_file_path(save_dir:str):
  return os.path.join(save_dir, "codepoints_cluster_map.pkl")


class AbstractFeatureExtractor:
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


class AbstractFeatureClusterer:
  def __init__(self, save_dir:str):
    self.sd = save_dir
    with open(generate_features_dict_file_path(self.sd), 'rb') as f:
      self.features_dict = pickle.load(f)

  def _cluster_features_into_equivalence_classes(self, features_dict:dict) -> (dict, dict):
    """
    *
    :param features_dict: keys are codepoint integers, values are numpy arrays of identical dimension
    :return: mapping of codepoints to cluster id, a mapping of cluster ids to lists of codepoints
    """
    raise NotImplementedError

  def find_and_save_equivalence_classes(self):
    codepoints_cluster_map, cluster_codepoints_map = self._cluster_features_into_equivalence_classes(self.features_dict)
    with open(generate_codepoints_cluster_map_file_path(self.sd), 'wb+') as f:
      pickle.dump(codepoints_cluster_map, f)
    with open(generate_cluster_codepoints_map_file_path(self.sd), 'wb+') as f:
      pickle.dump(cluster_codepoints_map, f)
