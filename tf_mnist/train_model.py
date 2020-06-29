import efficientnet.keras as efn
import numpy as np
import tf_mnist.generate_datasets as gd
import cv2 as cv
import tensorflow as tf
import tensorflow.keras as K

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
    # Epsilon included for numerical stability
    x_ = (tf.sqrt(einsum(x1, x1)) * tf.sqrt(einsum(x2, x2))) + (10e-5)
    return einsum(x1, x2) / x_


# Where x1 is an anchor input, x2 belongs to the same class and x3 belongs to a different class
def cos_triplet_loss(x1, x2, x3):
    return (tf.reduce_mean((cos_sim(x1, x3) - cos_sim(x1, x2))) + 1) / 2


def euc_triplet_loss(x1, x2, x3):
    return tf.math.maximum(0, tf.reduce_mean(tf.norm((x1 - x2) + 1e-10) - tf.norm((x1 - x3) + 1e-10)) + 80)


def choose_model(choice):
    if choice == EFFICIENTNET:
        # Pooling options: max, avg, none
        return efn.EfficientNetB4(weights='imagenet', input_tensor=tf.keras.layers.Input([28, 28, 3]),
                                  include_top=False, pooling='avg')
    elif choice == CUSTOM:
        model = K.models.Sequential()
        model.add(K.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                                  activation='relu',
                                  input_shape=(28, 28, 1)))
        model.add(K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(K.layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(K.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(30, activation='sigmoid'))
        return model


def unison_shuffled_copies(a, b, c):
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

def train():
    # Model Choices
    model = choose_model(CUSTOM)

    # Training Settings
    optimizer = tf.keras.optimizers.Adam()
    n_epochs = 8
    training_choices = [GRAY, NORMALIZE]
    train_batch_size = 128
    test_batch_size = 540
    n_train_batches = 54210 // train_batch_size
    n_test_batches = 17820 // test_batch_size

    # Training Loop
    for epoch in range(n_epochs):
        anchors, positives, negatives, x_1_test, x_2_test, y_test = gd.compile_triplet_datasets()
        anchors, positives, negatives = unison_shuffled_copies(anchors, positives, negatives)
        anchors_converted = data_preprocess(anchors, training_choices[0], training_choices[1])
        positives_converted = data_preprocess(positives, training_choices[0], training_choices[1])
        negatives_converted = data_preprocess(negatives, training_choices[0], training_choices[1])
        x1_test_converted = data_preprocess(x_1_test, training_choices[0], training_choices[1])
        x2_test_converted = data_preprocess(x_2_test, training_choices[0], training_choices[1])
        y_test = tf.squeeze(tf.convert_to_tensor(y_test, dtype=tf.float32))
        epoch_train_loss = tf.convert_to_tensor(0, dtype=tf.float32)
        for train_batch in range(n_train_batches):
            with tf.GradientTape() as tape:
                batch_start = train_batch_size * train_batch
                anc = anchors_converted[batch_start:batch_start + train_batch_size]
                pos = positives_converted[batch_start:batch_start + train_batch_size]
                neg = negatives_converted[batch_start:batch_start + train_batch_size]
                anchor_forward, positive_forward, negative_forward = model(anc), model(pos), model(neg)
                loss = cos_triplet_loss(anchor_forward, positive_forward, negative_forward)
                epoch_train_loss += loss
            # Get gradients of loss wrt the weights.
            gradients = tape.gradient(loss, model.trainable_weights)
            # Update the weights of the model.
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        epoch_test_loss = tf.convert_to_tensor(0, dtype=tf.float32)
        for test_batch in range(n_test_batches):
            batch_start = test_batch_size * test_batch
            x1_forward, x2_forward = model(x1_test_converted[batch_start:batch_start + test_batch_size]), model(
                x2_test_converted[batch_start:batch_start + test_batch_size])
            test_cos_sim = cos_sim(x1_forward, x2_forward)
            test_labels = y_test[batch_start:batch_start + test_batch_size]
            cos_sim_as_actual = tf.math.abs(test_labels - test_cos_sim)
            test_batch_loss = tf.reduce_mean(cos_sim_as_actual)
            epoch_test_loss += test_batch_loss
        avg_epoch_train_loss = (epoch_train_loss / n_train_batches).numpy()
        avg_epoch_test_loss = (epoch_test_loss / n_test_batches).numpy()
        print(f"Epoch #{epoch + 1} Training Loss: " + str(avg_epoch_train_loss))
        print(f"Epoch #{epoch + 1} Testing Loss: " + str(avg_epoch_test_loss))
        print()

    # serialize model to JSON
    model_json = model.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model/model.h5")

if __name__ == '__main__':
    train()