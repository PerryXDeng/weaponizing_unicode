import tensorflow as tf

# for training usage
BATCH_SIZE = 32
EPOCHS_PER_VALIDATION = 1
VALIDATION_BATCH_SIZE = 48
NUM_EPOCHS = 5
# regularization
DROP = False
L2 = True
DROP_RATE = 0.2
LAMBDA = 0.005
# for customizing momentum optimizer
BASE_LEARNING_RATE = 0.001
DECAY_RATE = 0.95
MOMENTUM = 0.9

# data type used in matrices
DTYPE = tf.float32

# dims for input
DATA_FORMAT = ('NCHW' if tf.test.is_built_with_cuda() else 'NHWC')
IMG_X = 28
IMG_Y = IMG_X
NUM_CHANNELS = 1
INPUT_SHAPE = ([BATCH_SIZE, NUM_CHANNELS, IMG_X, IMG_Y] if DATA_FORMAT == 'NCHW'
               else [BATCH_SIZE, IMG_X, IMG_Y, NUM_CHANNELS])

# dims for twin network
FILTER_LEN = 3
NUM_FILTERS = [64, 128, 256, 256]
NUM_CONVS = len(NUM_FILTERS)
POOL_KLEN = 2
POOL_KDIM = ([1, POOL_KLEN, POOL_KLEN, 1] if DATA_FORMAT == 'NHWC'
             else [1, 1, POOL_KLEN, POOL_KLEN])
NUM_POOL = 3
FEATURE_MAP_SIZE = ((IMG_X // (POOL_KLEN ** NUM_POOL)) ** 2) * NUM_FILTERS[-1]
NUM_FC_NEURONS = 128 # decides the number of features before squared diff


# for discrete prediction
THRESHOLD = 0.5

# constants for normalization of grayscaled inputs into (-0.5, 0.5)
# Xnorm = (X - (RANGE / 2)) / RANGE
PIXEL_RANGE = 255

# logging
log_dir = "/tmp/siamese_tensorflow/test"

# for fake data generator
SEED = 42069
