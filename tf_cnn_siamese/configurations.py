import tensorflow as tf

# for training usage, exponential decay with momentum gradients
DROP_RATE = 0.5
BATCH_SIZE = 48 # about 3.1 G of float32 per 64 sized batch
LAMBDA = 5e-4
BASE_LEARNING_RATE = 0.001
DECAY_RATE = 0.95
MOMENTUM = 0.9
EPOCHS_PER_VALIDATION = 1
NUM_EPOCHS = 5

# data type used in matrices
DTYPE = tf.float32

# dims for input
DATA_FORMAT = ('NCHW' if tf.test.is_built_with_cuda() else 'NWHC')
IMG_X = 28
IMG_Y = 28
NUM_CHANNELS = 1
INPUT_SHAPE = ([BATCH_SIZE, NUM_CHANNELS, IMG_X, IMG_Y] if DATA_FORMAT == 'NCHW'
               else [BATCH_SIZE, IMG_X, IMG_Y, NUM_CHANNELS])

# for discrete prediction
THRESHOLD = 0.5

# constants for normalization of grayscaled inputs into (-0.5, 0.5)
# Xnorm = (X - (RANGE / 2)) / RANGE
PIXEL_RANGE = 255

# logging
log_dir = "/tmp/siamese_tensorflow/test"

# for fake data generator
SEED = 42069
