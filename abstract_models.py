class AbstractFeatureExtractor:
  def __init__(self, model_path:str, batch_size:int, features_save_path:str):
    self.model = self._restore_model(model_path)
    self.bs = batch_size
    self.feature_save_path = features_save_path

  def _restore_model(self, model_path:str):
    raise NotImplementedError

  def _init_test_data_loader(self, batch_size:int):
    raise NotImplementedError

  def _batch_extract_and_save_features(self, data_loader,features_save_path):
    raise NotImplementedError

  def extract_and_save_features(self):
    data_loader = self._init_test_data_loader(self.bs)
    self._batch_extract_and_save_features(data_loader, self.feature_save_path)


class AbstractFeatureClusterer:
  def __init__(self):
    pass

  def cluster_features_into_equivalence_classes(self):
    raise NotImplementedError
