import pickle
import numpy as np
import cupy as cp
from unicode_info.database import generate_supported_consortium_feature_vectors_and_clusters_dict
from feature_cluster_algos import load_sim_mat_float16

def find_indices_of_homoglyphs_gpu(sim_mat:np.ndarray, threshold:float, batch_size:int=10000):
  homoglyph_indices = []
  n, _ = sim_mat.shape
  num_batches = n // batch_size
  for batch_i in range(num_batches):
    start = batch_i * batch_size
    end = (batch_i + 1) * batch_size
    if end > n: end = n
    homoglyph_indices.append(cp.asnumpy(cp.nonzero(cp.sum((cp.asarray(sim_mat[start:end]) > threshold), axis=1)-1)[0]) + start)
  return set(np.concatenate(homoglyph_indices))


def find_unknown_homoglyphs(sim_mat:np.ndarray, threshold:float, batch_size:int, known_indices:set, indices_to_codepoint:list):
  found_indices = find_indices_of_homoglyphs_gpu(sim_mat, threshold, batch_size)
  unknown_indices = found_indices.difference(known_indices)
  get_homoglyph_codepoints = lambda index: [indices_to_codepoint[homo] for homo in np.nonzero(sim_mat[index]>threshold)[0]]
  unknown_codepoints_sets_dict = {indices_to_codepoint[index]:get_homoglyph_codepoints(index) for index in unknown_indices}
  # one entry for every homoglyph, one set of homoglhphs for every entry
  # NOT equivalence classes, so many of these sets have overlap
  return unknown_codepoints_sets_dict


def save_filtered_simmat_to_simmap():
  consortium_codepoint_vectors_dict, _ = generate_supported_consortium_feature_vectors_and_clusters_dict(9999, 'features_dict_file.pkl')
  consortium_codepoints = set(consortium_codepoint_vectors_dict.keys())
  unicode_codepoint_vectors_dict = pickle.load(open('features_dict_file.pkl', 'rb'))
  unicode_index_codepoint_map = list(unicode_codepoint_vectors_dict.keys())
  unicode_codepoint_index_map = {codepoint:index for index, codepoint in enumerate(unicode_index_codepoint_map)}
  consortium_indices = {unicode_codepoint_index_map[codepoint] for codepoint in consortium_codepoints}
  sim_mat = load_sim_mat_float16()
  unknown_codepoints = find_unknown_homoglyphs(sim_mat, threshold=0.9, batch_size=10000, known_indices=consortium_indices, indices_to_codepoint=unicode_index_codepoint_map)
  unknown_codepoints_simvec_map = {codepoint:sim_mat[unicode_codepoint_index_map[codepoint]] for codepoint in unknown_codepoints.keys()}
  with open('simmap_pointninethreshold.pkl', 'wb+') as f:
    pickle.dump(unknown_codepoints_simvec_map, f)

if __name__ == "__main__":
  save_filtered_simmat_to_simmap()
