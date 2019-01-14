from fake_siamese import hyperparameters as hp
from fake_siamese import fakedata as fd
from fake_siamese import numpy_backpropagation as nb

import numpy as np

def randinit_weights(twin_weights, joined_weights, twin_bias, joined_bias):
  for i in range(1, hp.TWIN_L):
    dim = (hp.TWIN_NET[i], hp.TWIN_NET[i - 1])
    twin_weights[i - 1] = np.random.rand(dim[0], dim[1]) - 0.5
    twin_bias[i - 1] = np.random.rand(dim[0]) - 0.5
  for i in range(1, hp.JOINED_L):
    dim = (hp.JOINED_NET[i], hp.JOINED_NET[i - 1])
    joined_weights[i - 1] = np.random.rand(dim[0], dim[1]) - 0.5
    joined_bias[i - 1] = np.random.rand(dim[0]) - 0.5

def main():
  (x1, x2, y) = fd.generate_x1_x2_y()
  twin_weights = np.ndarray(hp.TWIN_L - 1, dtype=np.matrix)
  twin_bias = np.ndarray(hp.TWIN_L - 1, dtype=np.ndarray)
  joined_weights = np.ndarray(hp.JOINED_L - 1, dtype=np.matrix)
  joined_bias = np.ndarray(hp.JOINED_L - 1, dtype=np.ndarray)

  randinit_weights(twin_weights, joined_weights, twin_bias, joined_bias)

  (cost, twin_weights_gradients, joined_weight_gradients) = \
      nb.cost_derivatives(x1, x2, y, twin_weights, twin_bias,
                          joined_weights, joined_bias)

  (numerical_twin_gradients, numerical_joined_gradients) = \
      nb.num_approx_aggregate(x1, x2, twin_weights, twin_bias,
                              joined_weights, joined_bias)
  print("\nnp cost")
  print(cost)
  print("\nnp twin weights gradients")
  print(twin_weights_gradients)
  print("\napproximated twin gradients")
  print(numerical_twin_gradients)
  print("\nnp joined weights gradients")
  print(joined_weight_gradients)
  print("\napproximated joined gradients")
  print(numerical_joined_gradients)

if __name__ == "__main__":
  main()
