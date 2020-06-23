import efficientnet.keras as efn
import numpy as np
import tf_mnist.generate_datasets as gd
import cv2 as cv
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations


def einsum(a, b):
    return tf.einsum('ji,ji->i', a, b)


def cos_sim(x1, x2):
    return einsum(x1, x2) / (tf.sqrt(einsum(x1, x1)) * tf.sqrt(einsum(x2, x2)))


# Where x1 is an anchor input, x2 belongs to the same class and x3 belongs to a different class
def cos_triplet_loss(x1, x2, x3):
    return (tf.reduce_mean((cos_sim(x1, x3) - cos_sim(x1, x2))) + 2) / 4


def make_prediction(model, input):
    converted = cv.cvtColor(input, cv.COLOR_GRAY2RGB)
    x = np.expand_dims(converted, 0)
    y = model.predict(x)
    return y


# Pooling options: max, avg, none
model = efn.EfficientNetB4(weights='imagenet', input_tensor=tf.keras.layers.Input([28, 28, 3]), include_top=False,
                           pooling='avg')

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
batch_size = 10
n_epochs = 100
for epoch in range(n_epochs):
    anchors, positives, negatives, x_1_test, x_2_test, y_test = gd.compile_triplet_datasets()
    anchors_converted = np.asarray([cv.cvtColor(v, cv.COLOR_GRAY2RGB) for v in anchors])
    positives_converted = np.asarray([cv.cvtColor(v, cv.COLOR_GRAY2RGB) for v in positives])
    negatives_converted = np.asarray([cv.cvtColor(v, cv.COLOR_GRAY2RGB) for v in negatives])
    anchors_converted = (anchors_converted / 255)
    positives_converted = (positives_converted / 255)
    negatives_converted = (negatives_converted / 255)
    for batch in range(54210 // batch_size):
        with tf.GradientTape() as tape:
            batch_start = batch_size * batch
            anc = anchors_converted[batch_start:batch_start + batch_size]
            pos = positives_converted[batch_start:batch_start + batch_size]
            neg = negatives_converted[batch_start:batch_start + batch_size]
            anchor_forward, positive_forward, negative_forward = model(anc), model(pos), model(neg)
            print(anchor_forward)
            loss = cos_triplet_loss(anchor_forward, positive_forward, negative_forward)
            print(loss)
        # Get gradients of loss wrt the weights.
        gradients = tape.gradient(loss, model.trainable_weights)
        # Update the weights of the model.
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
