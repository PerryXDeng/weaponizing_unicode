import argparse
import numpy as np
import cv2 as cv
import tensorflow as tf
import tensorflow.keras as K
from generate_datasets import compile_datasets
import efficientnet.keras as efn
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument('-trsi', '--training_set_iterations', action='store', type=int, default=6)
parser.add_argument('-trss', '--training_set_size', action='store', type=int, default=500)
parser.add_argument('-tess', '--testing_set_size', action='store', type=int, default=2000)
parser.add_argument('-bs', '--batch_size', action='store', type=int, default=64)

# Type of global pooling applied to the output of the last convolutional layer, giving a 2D tensor
# Options: max, avg (None also an option, probably not something we want to use)
parser.add_argument('-p', '--pooling', action='store', type=str, default='avg')

parser.add_argument('-lr', '--learning_rate', action='store', type=float, default=.001)

# Vector comparison method
# Options: cos, euc
parser.add_argument('-lf', '--loss_function', action='store', type=str, default='cos')

parser.add_argument('-s', '--save_model', action='store', type=bool, default=False)
parser.add_argument('-img', '--img_size', action='store', type=int, default=100)
parser.add_argument('-font', '--font_size', action='store', type=float, default=.4)
parser.add_argument('-e', '--epsilon', action='store', type=float, default=10e-5)
args = parser.parse_args()


# DEPRECATED
def einsum(a, b):
    return tf.einsum('ij,ij->i', a, b)


# DEPRECATED
def cos_sim_einsum(x1, x2):
    # Epsilon included for numerical stability
    x_ = (tf.sqrt(einsum(x1, x1)) * tf.sqrt(einsum(x2, x2))) + args.epsilon
    return einsum(x1, x2) / x_


def cos_sim(x1, x2):
    axis_1 = x1.shape[0]
    axis_2 = x2.shape[1]
    a_v = tf.reshape(x1, [axis_1, 1, axis_2])
    b_v = tf.reshape(x2, [axis_1, axis_2, 1])
    return tf.reshape(tf.matmul(a_v, b_v), [axis_1]) / ((tf.norm(x1, axis=1) * tf.norm(x2, axis=1)) + args.epsilon)


# Where x1 is an anchor input, x2 belongs to the same class and x3 belongs to a different class
def cos_triplet_loss(x1, x2, x3):
    return (tf.reduce_mean((cos_sim(x1, x3) - cos_sim(x1, x2))) + 2) / 4


# Where x1 is an anchor input, x2 belongs to the same class and x3 belongs to a different class
def euc_triplet_loss(x1, x2, x3, c=1000):
    return tf.reduce_mean(tf.norm(x1 - x2) - tf.norm(x1 - x3))


def data_preprocess(data):
    return tf.convert_to_tensor((data - (255 / 2)) / (255 / 2), dtype=tf.float32)
    

def train(loss_function):
    model = efn.EfficientNetB5(weights='imagenet',
                               input_tensor=tf.keras.layers.Input([args.img_size, args.img_size, 3]), include_top=False,
                               pooling=args.pooling)
    # Training Settings
    optimizer = tf.keras.optimizers.Adam()
    training_iterations = args.training_set_size // args.batch_size
    # Training Loop
    for epoch in range(args.training_set_iterations):
        print("Processing data...")
        anchors, positives, negatives, x1_test, x2_test, y_test = compile_datasets(args.training_set_size,
                                                                                   1,
                                                                                   font_size=args.font_size,
                                                                                   img_size=args.img_size,
                                                                                   color_format='RGB')
        anchors, positives, negatives = data_preprocess(anchors), data_preprocess(
            positives), data_preprocess(negatives)
        print("Done")
        epoch_train_loss = 0
        for train_batch in range(training_iterations):
            with tf.GradientTape() as tape:
                batch_start = args.batch_size * train_batch
                anc = anchors[batch_start:batch_start + args.batch_size]
                pos = positives[batch_start:batch_start + args.batch_size]
                neg = negatives[batch_start:batch_start + args.batch_size]
                anchor_forward, positive_forward, negative_forward = model(anc), model(pos), model(neg)
                loss = loss_function(anchor_forward, positive_forward, negative_forward)
                print(loss.numpy())
                epoch_train_loss += loss.numpy()
            # Get gradients of loss wrt the weights.
            gradients = tape.gradient(loss, model.trainable_weights)
            # Update the weights of the model.
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        avg_epoch_train_loss = (epoch_train_loss / training_iterations)
        print(f"Epoch #{epoch + 1} Training Loss: " + str(avg_epoch_train_loss))
        print()

    if(args.save_model):
      # serialize model to JSON
      model_json = model.to_json()
      with open("model/model.json", "w") as json_file:
          json_file.write(model_json)
      # serialize weights to HDF5
      model.save_weights("model/model.h5")
    
    anchors, positives, negatives, x1_test, x2_test, y_test = compile_datasets(1,
                                                                           args.testing_set_size,
                                                                           font_size=args.font_size,
                                                                           img_size=args.img_size,
                                                                           color_format='RGB')
    x1_test, x2_test = data_preprocess(x1_test), data_preprocess(x2_test)
    x1_test_forward, x2_test_forward = model(x1_test,training=False), model(x2_test,training=False)
    test_cos_sim = (cos_sim(x1_test_forward,x2_test_forward).numpy()).reshape(-1,1)
    prediction_model = LogisticRegression(random_state=0).fit(test_cos_sim, y_test)
    print(prediction_model.score(test_cos_sim,y_test))



if __name__ == '__main__':
    if args.loss_function == 'cos':
        loss_function = cos_triplet_loss
    elif args.loss_function == 'euc':
        loss_function = euc_triplet_loss
    train(loss_function=loss_function)
