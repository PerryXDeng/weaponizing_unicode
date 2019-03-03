import tensorflow as tf

# for random generator
SEED = 42069

# dims for input
IMG_X = 28
IMG_Y = 28
NUM_CHANNELS = 1

# constants for normalization of grayscaled inputs into (-0.5, 0.5)
# Xnorm = (X - (RANGE / 2)) / RANGE
PIXEL_RANGE = 255

# data type used in matrices
DTYPE = tf.float32

# for training usage, exponential decay with momentum gradients
DROP_RATE = 0.5
BATCH_SIZE = 64
LAMBDA = 5e-4
BASE_LEARNING_RATE = 0.01
DECAY_RATE = 0.95
MOMENTUM = 0.9
VALIDATION_INTERVAL = 1
