import tensorflow as tf

# dims for input
IMG_X = 28
IMG_Y = 28

# constants for normalization of grayscaled inputs into (-0.5, 0.5)
# Xnorm = (X - (RANGE / 2)) / RANGE
PIXEL_RANGE = 255

# data type used in matrices
DTYPE = tf.float32

# for training usage
DROP_RATE = 0.5
