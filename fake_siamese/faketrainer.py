from fake_siamese import hyperparameters as hp
from fake_siamese import fakedata as fd
from fake_siamese import numpy_backpropagation as nb

import numpy as np

def randinit_weights(twin_weights, diff_weights):
  for i in range(1, hp.TWIN_L):
    dim = (hp.TWIN_NET[i], hp.TWIN_NET[i - 1])
    twin_weights[i - 1] = np.random.rand(dim[0], dim[1]) - 0.5
    twin_bias[i - 1] = np.random.rand(dim[0]) - 0.5
  for i in range(1, hp.DIFF_L):
    dim = (hp.DIFF_NET[i], hp.DIFF_NET[i - 1])
    diff_weights[i - 1] = np.random.rand(dim[0], dim[1]) - 0.5
    diff_bias[i - 1]] = np.random.rand(dim[0]) - 0.5

def main():
  (x1, x2, y) = fd.generate_x1_x2_y()
  twin_weights = np.ndarray(hp.TWIN_L - 1, dtype=np.matrix)
  twin_bias = np.ndarray(hp.TWIN_L - 1, dtype=np.ndarray)
  diff_weights = np.ndarray(hp.DIFF_L - 1, dtype=np.matrix)
  diff_bias = np.ndarray(hp.DIFF_L - 1, dtype=np.ndarray)

  randinit_weights(twin_weights, diff_weights, twin_bias, diff_bias)

  (cost, twin_weights_gradients, twin_bias_gradients,
      diff_weight_gradients, diff_bias_gradients) = \
      nb.cost_gradients(x1, x2, y, twin_weights, twin_bias,
                        diff_weights, diff_bias)
  print("\nnp cost")
  print(cost)
  print("\nnp twin weights gradients")
  print(twin_weights_gradients)
  print("\nnp twin bias gradients")
  print(twin_bias_gradients)
  print("\nnp diff weights gradients")
  print(diff_weight_gradients)

if __name__ == "__main__":
  main()
