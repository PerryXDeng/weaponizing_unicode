from fake_siamese import hyperparameters as hp
from fake_siamese import fakedata as fd
from fake_siamese import numpy_backpropagation as nb

import numpy as np

def main():
  (x1, x2, y) = fd.generate_x1_x2_y()
  twin_weights = np.ndarray(hp.TWIN_L - 1, dtype=np.matrix)
  twin_bias = np.ndarray(hp.TWIN_L - 1, dtype=np.float)
  diff_weights = np.ndarray(hp.DIFF_L - 1, dtype=np.matrix)
  diff_bias = np.ndarray(hp.DIFF_L - 1, dtype=np.float)

  for i in range(1, hp.TWIN_L):
    twin_weights[i - 1] = np.matrix(
        np.zeros((hp.TWIN_NET[i], hp.TWIN_NET[i-1])))
  for i in range(1, hp.DIFF_L):
    diff_weights[i - 1] = np.matrix(
        np.zeros((hp.DIFF_NET[i], hp.DIFF_NET[i-1])))

  (cost, twin_weights_gradients, twin_bias_gradients,
      diff_weight_gradients, diff_bias_gradients) = \
      nb.backpropagation(x1, x2, y, twin_weights, twin_bias,
                         diff_weights, diff_bias)
  print("\ncost")
  print(cost)
  print("\ntwin weights gradients")
  print(twin_weights_gradients)
  print("\ntwin bias gradients")
  print(twin_bias_gradients)
  print("\ndiff weights gradients")
  print(diff_weight_gradients)

if __name__ == "__main__":
  main()