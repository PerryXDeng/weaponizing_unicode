from fake_siamese import hyperparameters as hp
import numpy as np

def generate_x1_x2_y():
  x_dim = (hp.SAMPLE_SIZE, hp.FEATURE_SIZE)
  y_dim = hp.SAMPLE_SIZE
  return np.random.rand(x_dim[0], x_dim[1]) - 0.5, \
         np.random.rand(x_dim[0], x_dim[1]) - 0.5, \
         np.random.choice(a=[0, 1], size=y_dim, p=[0.49, 0.51]) # rand booleans
