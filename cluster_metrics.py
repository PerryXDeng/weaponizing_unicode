from unicode_info.database import download_and_parse_unicode_clusters


def calculate_mean_best_case_coverage(predicted_codepoints_clusters_map:dict,
                                      predicted_clusters_codepoints_map:dict) -> float:
  """
  "coverage"
  :param predicted_clusters_codepoints_map: mapping of codepoints to cluster id
  :param predicted_codepoints_clusters_map: mapping of cluster ids to lists of codepoints
  :return:
  """
  unicode_clusters_codepoints_map = download_and_parse_unicode_clusters()
  raise NotImplementedError
