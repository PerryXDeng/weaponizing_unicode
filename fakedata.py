import hyperparameters as hp
import numpy as np

def generate_x1_x2_y():
  x_dim = (hp.SAMPLE_SIZE, hp.FEATURE_SIZE)
  y_dim = hp.SAMPLE_SIZE
  return np.random.rand(x_dim), np.random.rand(x_dim), np.random.rand(y_dim)
