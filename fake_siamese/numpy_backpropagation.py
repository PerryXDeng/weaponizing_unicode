from fake_siamese import hyperparameters as hp
import numpy as np

def sigmoid(z):
  return 1. / (1. + np.exp(-z))

def binary_cross_entropy(h, y):
  return -y * np.log(h) + (1 - y) * np.log(1 - h)

def activation(prev, weights, bias):
  prev_copy = np.r_[np.ones(prev.shape[1])[np.newaxis], prev]
  weights_copy = np.c_[bias, weights]
  return sigmoid(np.matmul(weights_copy, prev_copy))

def numpy_feedforward(x_1, x_2, twin_weights, joined_weights,
                      twin_bias, joined_bias):
  # activation value matrices of the two twin networks and the joined network
  a_1 = np.ndarray(hp.TWIN_L, dtype=np.matrix)
  a_2 = np.ndarray(hp.TWIN_L, dtype=np.matrix)
  a_d = np.ndarray(hp.JOINED_L, dtype=np.matrix)

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
  for i in range(1, hp.JOINED_L):
    a_d[i] = activation(a_d[i - 1], joined_weights[i - 1], joined_bias[i - 1])

  return a_1, a_2, a_d

def regularize(weights, bias, gradients, layers):
  for n in range(1, layers):
    regularization_offset = hp.REG_CONST \
        * np.concatenate((bias[n - 1], weights[n - 1]), axis=1)
    gradients[n - 1] += regularization_offset
    gradients[n - 1][0] -= regularization_offset[0]

def cost_gradients(x_1, x_2, y, twin_weights, twin_bias,
                   joined_weights, joined_bias):
  # zero initializes cost and gradients
  modelcost = np.float(0)
  twin1_transformations_derivatives = np.ndarray(hp.TWIN_L - 1, dtype=np.ndarray)
  twin2_transformations_derivatives = np.ndarray(hp.TWIN_L - 1, dtype=np.ndarray)
  twin_weights_gradients = np.ndarray(hp.TWIN_L - 1, dtype=np.matrix)
  joined_transformations_derivatives = np.ndarray(hp.JOINED_L - 1, dtype=np.ndarray)
  joined_weights_gradients = np.ndarray(hp.JOINED_L - 1, dtype=np.matrix)
  for i in range(1, hp.TWIN_L):
    twin_weights_gradients[i - 1] = np.matrix(
        np.zeros((hp.TWIN_NET[i], hp.TWIN_NET[i-1] + 1)))
  for i in range(1, hp.JOINED_L):
    joined_weights_gradients[i - 1] = np.matrix(
        np.zeros((hp.JOINED_NET[i], hp.JOINED_NET[i - 1] + 1)))
  #twin_bias_gradients.fill(0.)
  #joined_bias_gradients.fill(0.)

  # sums up the derivatives of cost for each sample 
  for i in range(0, hp.SAMPLE_SIZE):
    (a_1, a_2, a_d) = numpy_feedforward(x_1[i], x_2[i], twin_weights,
                                        joined_weights, twin_bias, joined_bias)
    modelcost += binary_cross_entropy(a_d[hp.JOINED_L - 1], y[i])
    
    joined_transformations_derivatives[hp.JOINED_L - 2] = \
        a_d[hp.JOINED_L - 1] - y[i]
    for n in reversed(range(0, hp.JOINED_L - 2)):
      joined_transformations_derivatives[n] = \
          np.matmul(joined_weights[n + 1].T, joined_transformations_derivatives[n + 1]) \
          * (a_d[n + 1] * (1 - a_d[n + 1]))
    twin1_transformations_derivatives[hp.TWIN_L - 2] = \
        2 * (a_1[hp.TWIN_L - 1] - a_2[hp.TWIN_L - 1])
    twin2_transformations_derivatives[hp.TWIN_L - 2] = \
        -1 * twin1_transformations_derivatives[hp.TWIN_L - 2]
    for n in reversed(range(0, hp.TWIN_L - 2)):
      twin1_transformations_derivatives[n] = \
          np.matmul(twin_weights[n + 1].T, twin1_transformations_derivatives[n + 1]) \
          * (a_1[n + 1] * (1 - a_1[n + 1]))
      twin2_transformations_derivatives[n] = \
          np.matmul(twin_weights[n + 1].T, twin2_transformations_derivatives[n + 1]) \
          * (a_2[n + 1] * (1 - a_2[n + 1]))
    
    for n in range(1, hp.JOINED_L):
      ad_concat_1 = np.r_[np.ones(a_d[n - 1].shape[1])[np.newaxis], a_d[n - 1]]
      joined_weights_gradients[n - 1] += \
          np.matmul(joined_transformations_derivatives[n - 1], ad_concat_1.T)
    for n in range(1, hp.TWIN_L):
      a1_concat_1 = np.r_[np.ones(a_1[n - 1].shape[1])[np.newaxis], a_1[n - 1]]
      a2_concat_1 = np.r_[np.ones(a_2[n - 1].shape[1])[np.newaxis], a_2[n - 1]]
      twin_weights_gradients[n - 1] += \
          np.add(np.matmul(twin1_transformations_derivatives[n - 1], a1_concat_1.T)
          , np.matmul(twin2_transformations_derivatives[n - 1], a2_concat_1.T))

  # take their mean
  modelcost /= hp.SAMPLE_SIZE
  for n in range(1, hp.JOINED_L):
    joined_weights_gradients[n - 1] = \
        joined_weights_gradients[n - 1] / hp.SAMPLE_SIZE
  for n in range(1, hp.TWIN_L):
    twin_weights_gradients[n - 1] = twin_weights_gradients[n - 1] \
                                    / hp.SAMPLE_SIZE

  #regularize(twin_weights, twin_bias, twin_weights_gradients, hp.TWIN_L)
  #regularize(deep_weights, deep_bias, deep_weights_gradients, hp.DEEP_L)
  return modelcost, twin_weights_gradients, joined_weights_gradients

def numerical_derivative_approximation(twin_weights, twin_bias, 
      joined_weights, joined_bias, i, j, l, joined=True):
  twin_weights_copy_1 = np.ndarray(hp.TWIN_L - 1, dtype=np.matrix)
  twin_bias_copy_1 = np.ndarray(hp.TWIN_L - 1, dtype=np.ndarray)
  joined_weights_copy_1 = np.ndarray(hp.JOINED_L - 1, dtype=np.matrix)
  joined_bias_copy_1 = np.ndarray(hp.JOINED_L - 1, dtype=np.ndarray)

  twin_weights_copy_2 = np.ndarray(hp.TWIN_L - 1, dtype=np.matrix)
  twin_bias_copy_2 = np.ndarray(hp.TWIN_L - 1, dtype=np.ndarray)
  joined_weights_copy_2 = np.ndarray(hp.JOINED_L - 1, dtype=np.matrix)
  joined_bias_copy_2 = np.ndarray(hp.JOINED_L - 1, dtype=np.ndarray)
  for i in range(1, hp.TWIN_L):
    twin_weights_copy_1[i - 1] = twin_weights[i - 1]
    twin_bias_copy_1[i - 1] = twin_bias[i - 1]
    twin_weights_copy_2[i - 1] = twin_weights[i - 1]
    twin_bias_copy_2[i - 1] = twin_bias[i - 1]
  for i in range(1, hp.JOINED_L):
    joined_weights_copy_1[i - 1] = joined_weights[i - 1]
    joined_bias_copy_1[i - 1] = joined_bias[i - 1]
    joined_weights_copy_2[i - 1] = joined_weights[i - 1]
    joined_bias_copy_2[i - 1] = joined_bias[i - 1]

  if joined:
    if j == 0:
      new_bias_1 = np.ndarray.copy(joined_bias_copy_1[l])
      new_bias_2 = np.ndarray.copy(joined_bias_copy_2[l])
      new_bias_1[i] -= hp.NUMERICAL_DELTA
      new_bias_2[i] += hp.NUMERICAL_DELTA
      joined_bias_copy_1[l] = new_bias_1
      joined_bias_copy_2[l] = new_bias_2
    else:
      return
