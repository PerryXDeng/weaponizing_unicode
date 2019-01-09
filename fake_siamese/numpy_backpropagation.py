from fake_siamese import hyperparameters as hp
import numpy as np

def sigmoid(z):
  return 1. / (1. + np.exp(-z))

def binary_cross_entropy(h, y):
  return -y * np.log(h) + (1 - y) * np.log(1 - h)

def activation(prev, weights, bias):
  prev_copy = np.concat((np.ones(hp.SAMPLE_SIZE), prev), axis=0)
  weights_copy = np.concat((bias, weights), axis=1)
  return sigmoid(np.matmul(weights_copy, prev_copy))

def numpy_feedforward(x_1, x_2, twin_weights, diff_weights,
                      twin_bias, diff_bias):
  # activation value matrices of the two twin networks and the joined network
  a_1 = np.ndarray(hp.TWIN_L, dtype=np.matrix)
  a_2 = np.ndarray(hp.TWIN_L, dtype=np.matrix)
  a_d = np.ndarray(hp.DIFF_L, dtype=np.matrix)

  # transposing (horizontal) 1D input vectors into feature vectors
  a_1[0] = x_1[np.newaxis].T
  a_2[0] = x_2[np.newaxis].T

  # forward propagation of twins
  for i in range(1, hp.TWIN_L):
    a_1[i] = activation(a_1[i - 1], twin_weights[i - 1], twin_bias[i - 1])
    a_2[i] = activation(a_2[i - 1], twin_weights[i - 1], twin_bias[i - 1])

  # element wise squared diffrence of two twin network becomes the joined input
  a_d[0] = np.square(a_1[hp.TWIN_L - 1] - a_2[hp.TWIN_L - 1])

  # forward propagation of the joined network
  for i in range(1, hp.DIFF_L):
    a_d[i] = activation(a_d[i - 1], diff_weights[i - 1], diff_bias[i - 1])

  return a_1, a_2, a_d

def cost_gradients(x_1, x_2, y, twin_weights, twin_bias,
                    diff_weights, diff_bias):
  # zero initializes cost and gradients
  cost = np.float(0)
  twin1_activations_derivatives = np.ndarray(hp.TWIN_L - 1, dtype=np.ndarray)
  twin2_activations_derivatives = np.ndarray(hp.TWIN_L - 1, dtype=np.ndarray)
  twin_weights_gradients = np.ndarray(hp.TWIN_L - 1, dtype=np.matrix)
  #twin_bias_gradients = np.ndarray(hp.TWIN_L - 1, dtype=np.float)
  diff_activations_derivatives = np.ndarray(hp.TWIN_L - 1, dtype=np.ndarray)
  diff_weights_gradients = np.ndarray(hp.DIFF_L - 1, dtype=np.matrix)
  #diff_bias_gradients = np.ndarray(hp.DIFF_L - 1, dtype=np.float)
  for i in range(1, hp.TWIN_L):
    twin_weights_gradients[i - 1] = np.matrix(
        np.zeros((hp.TWIN_NET[i], hp.TWIN_NET[i-1] + 1)))
  for i in range(1, hp.DIFF_L):
    diff_weights_gradients[i - 1] = np.matrix(
        np.zeros((hp.DIFF_NET[i], hp.DIFF_NET[i-1] + 1)))
  #twin_bias_gradients.fill(0.)
  #diff_bias_gradients.fill(0.)

  # sums up the derivatives of cost for each sample 
  for i in range(0, hp.SAMPLE_SIZE):
    (a_1, a_2, a_d) = numpy_feedforward(x_1[i], x_2[i], twin_weights,
                                        diff_weights, twin_bias, diff_bias)
    cost += binary_cross_entropy(a_d[hp.DIFF_L - 1], y[i])
    
    diff_activations_derivatives[hp.DIFF_L - 2] = a_d[hp.DIFF_L - 1] - y[i]
    for n in range(hp.DIFF_L - 3, 0, -1):
      diff_activations_derivatives[n] = \
          np.matmul(diff_weights[n].T, diff_activations_derivatives[n + 1]) \
          * (a_d[n] * (1 - a_d[n]))

    twin1_activations_derivatives[hp.TWIN_L - 2] = \
        2 * (a_1[hp.TWIN_L - 1] - a_2[hp.TWIN_L - 1])
    twin2_activations_derivatives[hp.TWIN_L - 2] = \
        -1 * twin1_activations_derivatives[hp.TWIN_L - 2]
    for n in range(hp.TWIN_L - 3, 0, -1):
      twin1_activations_derivatives[n] = \
          np.matmul(twin_weights[n].T, twin1_activations_derivatives[n + 1]) \
          * (a_1[n] * (1 - a_1[n]))
      twin2_activations_derivatives[n] = \
          np.matmul(twin_weights[n].T, twin2_activations_derivatives[n + 1]) \
          * (a_2[n] * (1 - a_2[n]))
    
    for n in range(1, hp.DIFF_L):
      diff_weights_gradients[n - 1] += \
          np.matmul(diff_activations_derivatives[n - 1], (a_d[n - 1])[np.newaxis].T)
    for n in range(1, hp.TWIN_L):
      twin_weights_gradients[n - 1] += \
          np.matmul(twin1_activations_derivatives[n - 1],
                    (a_1[n - 1])[np.newaxis].T) \
          + np.matmul(twin2_activations_derivatives[n - 1],
                      (a_2[n - 1])[np.newaxis].T)

  # take their mean
  cost = cost/m
  for n in range(1, hp.DIFF_L):
    diff_weights_gradients[n - 1] = diff_weights_gradients[n - 1] / hp.SAMPLE_SIZE

  for n in range(1, hp.TWIN_L):
    twin_weights_gradients[n - 1] = twin_weights_gradients[n - 1] / hp.SAMPLE_SIZE

  return cost, twin_weights_gradients, diff_weights_gradients
