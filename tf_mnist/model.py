import efficientnet.keras as efn
import numpy as np
import tf_mnist.generate_datasets as gd
import cv2 as cv
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import activations

# Model choices
EFFICIENTNET = 'efn'
CUSTOM = 'custom'

# Color space choices
GRAY = 'gray'
RGB = 'rgb'

# Preprocessing choices
NORMALIZE = 'norm'
SCALE = 'scale'
MEAN = .1307
STD = .3081



def einsum(a, b):
    return tf.einsum('ij,ij->i', a, b)

def cos_sim(x1, x2):
    x_ = (tf.sqrt(einsum(x1, x1)) * tf.sqrt(einsum(x2, x2))) + (10e-5)
    return einsum(x1, x2) / x_

# Where x1 is an anchor input, x2 belongs to the same class and x3 belongs to a different class
def cos_triplet_loss(x1, x2, x3):
    return (tf.reduce_mean((cos_sim(x1, x3) - cos_sim(x1, x2))) + 1) / 2

def euc_triplet_loss(x1, x2, x3):
    return tf.math.maximum(0,tf.reduce_mean(tf.norm((x1-x2) + 1e-10)-tf.norm((x1-x3)+ 1e-10))+10)

def choose_model(choice):
    if choice == EFFICIENTNET:
        # Pooling options: max, avg, none
        return efn.EfficientNetB4(weights='imagenet', input_tensor=tf.keras.layers.Input([28, 28, 3]), include_top=False,pooling='avg')
    elif choice == CUSTOM:
        model = K.models.Sequential()
        model.add(K.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                                  activation='relu',
                                  input_shape=(28,28,1)))
        model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(K.layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(K.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(10, activation='sigmoid'))
        return model

def unison_shuffled_copies(a, b,c):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

def data_preprocess(input, color_space, cleaning_mode):
    if color_space == RGB:
        data = tf.convert_to_tensor([cv.cvtColor(v, cv.COLOR_GRAY2RGB) for v in input], dtype=tf.float32)
    elif color_space == GRAY:
        data = tf.convert_to_tensor(input, dtype=tf.float32)
    data = data / 255
    if cleaning_mode == NORMALIZE:
        data = (data - MEAN) / STD
    return data


# Model Choices
model = choose_model(CUSTOM)

# Training Settings
optimizer = tf.keras.optimizers.Adam()
batch_size = 100
n_epochs = 100

# Training Loop
for epoch in range(n_epochs):
    anchors, positives, negatives, x_1_test, x_2_test, y_test = gd.compile_triplet_datasets()
    anchors,positives,negatives = unison_shuffled_copies(anchors,positives,negatives)
    anchors_converted = data_preprocess(anchors,GRAY,NORMALIZE)
    positives_converted = data_preprocess(positives,GRAY,NORMALIZE)
    negatives_converted = data_preprocess(negatives, GRAY, NORMALIZE)

    for batch in range(54210 // batch_size):
        with tf.GradientTape() as tape:
            batch_start = batch_size * batch
            anc = anchors_converted[batch_start:batch_start + batch_size]
            pos = positives_converted[batch_start:batch_start + batch_size]
            neg = negatives_converted[batch_start:batch_start + batch_size]
            anchor_forward, positive_forward, negative_forward = model(anc), model(pos), model(neg)
            print(anchor_forward)
            loss = cos_triplet_loss(anchor_forward, positive_forward, negative_forward)
            print(loss,batch)
        # Get gradients of loss wrt the weights.
        gradients = tape.gradient(loss, model.trainable_weights)
        # Update the weights of the model.
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
