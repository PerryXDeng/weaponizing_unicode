import numpy as np
import cupy as cp

def find_indices_of_homoglyphs_gpu(sim_mat:np.ndarray, threshold:float, batch_size:int=10000):
  homoglyph_indices = []
  n, _ = sim_mat.shape
  num_batches = n // batch_size
  for batch_i in range(num_batches):
    start = batch_i * batch_size
    end = (batch_i + 1) * batch_size
    if end > n: end = n
    homoglyph_indices.append(cp.asnumpy(cp.nonzero(cp.sum((cp.asarray(sim_mat[start:end]) > threshold), axis=1))[0]) + start)
  return set(np.concatenate(homoglyph_indices))

def find_unknown_homoglyphs(sim_mat:np.ndarray, threshold:float, batch_size):
  return
