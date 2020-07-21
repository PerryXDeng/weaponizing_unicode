import argparse
import numpy as np
import cv2 as cv
import tensorflow as tf
import tensorflow.keras as K
from tf_unicode.generate_datasets import compile_datasets
import efficientnet.keras as efn

parser = argparse.ArgumentParser()
parser.add_argument('-trsi', '--training_set_iterations', action='store', type=int, default=5)
parser.add_argument('-trss', '--training_set_size', action='store', type=int, default=500)
parser.add_argument('-tess', '--testing_set_size', action='store', type=int, default=1)
parser.add_argument('-bs', '--batch_size', action='store', type=int, default=8)

# Type of global pooling applied to the output of the last convolutional layer, giving a 2D tensor
# Options: max, avg (None also an option, probably not something we want to use)
parser.add_argument('-p', '--pooling', action='store', type=str, default='avg')

parser.add_argument('-lr', '--learning_rate', action='store', type=float, default=.01)

# Vector comparison method
# Options: cos, euc
parser.add_argument('-lf', '--loss_function', action='store', type=str, default='cos')

parser.add_argument('-s', '--save_model', action='store', type=bool, default=True)
parser.add_argument('-img', '--img_size', action='store', type=int, default=100)
parser.add_argument('-font', '--font_size', action='store', type=float, default=.4)
parser.add_argument('-e', '--epsilon', action='store', type=float, default=10e-5)
args = parser.parse_args()


def cos_sim(x1, x2, epsilon):
    axis_1 = x1.shape[0]
    axis_2 = x2.shape[1]
    a_v = tf.reshape(x1, [axis_1, 1, axis_2])
    b_v = tf.reshape(x2, [axis_1, axis_2, 1])
    return tf.reshape(tf.matmul(a_v, b_v), [axis_1]) / ((tf.norm(x1, axis=1) * tf.norm(x2, axis=1)) + epsilon)


# Where x1 is an anchor input, x2 belongs to the same class and x3 belongs to a different class
def cos_triplet_loss(x1, x2, x3):
    return (tf.reduce_mean((cos_sim(x1, x3) - cos_sim(x1, x2))) + 2) / 4


# Where x1 is an anchor input, x2 belongs to the same class and x3 belongs to a different class
def euc_triplet_loss(x1, x2, x3):
    return tf.reduce_mean(tf.norm(x1 - x2) - tf.norm(x1 - x3))


def data_preprocess(data):
    return tf.convert_to_tensor((data - (255 / 2)) / (255 / 2), dtype=tf.float32)


def train_for_num_step(model, optimizer, start_step, end_step):
  print()
  for epoch in range(args.training_set_iterations):
    print("Processing data...")
    anchors, positives, negatives, x1_test, x2_test, y_test = compile_datasets(args.training_set_size,
                                                                               args.testing_set_size,
                                                                               font_size=args.font_size,
                                                                               img_size=args.img_size,
                                                                               color_format='RGB')
    anchors, positives, negatives, x1_test, x2_test = data_preprocess(anchors), data_preprocess(
      positives), data_preprocess(negatives), data_preprocess(x1_test), data_preprocess(x2_test)
    print("Done")
    epoch_train_loss = 0
    for step in range(start_step, end_step):
      with tf.GradientTape() as tape:
        batch_start = args.batch_size * (step - start_step)
        anc = anchors[batch_start:batch_start + args.batch_size]
        pos = positives[batch_start:batch_start + args.batch_size]
        neg = negatives[batch_start:batch_start + args.batch_size]
        anchor_forward, positive_forward, negative_forward = model(anc), model(pos), model(neg)
        loss = loss_function(anchor_forward, positive_forward, negative_forward)
        epoch_train_loss += loss.numpy()
      # Get gradients of loss wrt the weights.
      gradients = tape.gradient(loss, model.trainable_weights)
      # Update the weights of the model.
      optimizer.apply_gradients(zip(gradients, model.trainable_weights), global_step=step)


def train(loss_function):
    model = efn.EfficientNetB4(weights='imagenet',
                               input_tensor=tf.keras.layers.Input([args.img_size, args.img_size, 3]), include_top=False,
                               pooling=args.pooling)
    # Training Settings
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    training_iterations = args.training_set_size // args.batch_size
    testing_iterations = args.testing_set_size // 200
    # Training Loop
    train_for_num_step(model, optimizer, 0, training_iterations)
    epoch_test_loss = 0
    # for test_batch in range(testing_iterations):
    #     batch_start = 200 * test_batch
    #     x1_forward, x2_forward = model(x1_test[batch_start:batch_start + 200]), model(
    #         x2_test[batch_start:batch_start + 200])
    #     test_cos_sim = cos_sim(x1_forward, x2_forward)
    #     test_labels = y_test[batch_start:batch_start + 200]
    #     cos_sim_as_actual = tf.math.abs(test_labels - test_cos_sim)
    #     test_batch_loss = tf.reduce_mean(cos_sim_as_actual)
    #     epoch_test_loss += test_batch_loss
    avg_epoch_train_loss = (epoch_train_loss / training_iterations)
    avg_epoch_test_loss = (epoch_test_loss / testing_iterations)
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
  if args.loss_function == 'cos':
    loss_function = cos_triplet_loss
  elif args.loss_function == 'euc':
    loss_function = euc_triplet_loss
  else:
    loss_function = None
  train(loss_function=loss_function)
