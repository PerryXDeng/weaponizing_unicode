import argparse
import numpy as np
import efficientnet.keras as efn
import numpy as np
import tf_mnist.generate_datasets as gd
import cv2 as cv
import tensorflow as tf
import tensorflow.keras as K


# Argument parsing for CMD
parser = argparse.ArgumentParser(description="Hyperparameters for training model")
parser.add_argument('-b', '--batch_size', action='store', type=int, default=128)
parser.add_argument('-e', '--epochs', action='store', type=int, default=10)
args = parser.parse_args()