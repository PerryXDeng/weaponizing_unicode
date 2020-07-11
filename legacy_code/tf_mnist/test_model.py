import legacy_code.tf_mnist.generate_datasets as gd
import tensorflow.keras as K
import tensorflow as tf
import legacy_code.tf_mnist.train_model as Utils
import numpy as np


def test_model():
    anchors, positives, negatives, x_1_test, x_2_test, y_test = gd.compile_triplet_datasets()
    x_1_test_model = Utils.data_preprocess(x_1_test, Utils.GRAY, Utils.NORMALIZE)
    x_2_test_model = Utils.data_preprocess(x_2_test, Utils.GRAY, Utils.NORMALIZE)
    y_test = tf.squeeze(tf.convert_to_tensor(y_test, dtype=tf.int32))
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = K.models.model_from_json(loaded_model_json)
    loaded_model.load_weights("model/model.h5")
    batch_size = 540
    confidence_threshold = .5
    correct = 0
    for i in range(17820 // batch_size):
        start = i * batch_size
        a = loaded_model.predict(x_1_test_model[start:start + batch_size])
        b = loaded_model.predict(x_2_test_model[start:start + batch_size])
        cos_sim = Utils.cos_sim(a, b)
        predictions = tf.cast(cos_sim > confidence_threshold, dtype=tf.int32)
        correct_predictions_np = tf.cast(predictions == y_test[start:start + batch_size], dtype=tf.int8).numpy()
        correct += np.sum(correct_predictions_np)
    print(f"Model accuracy, with a confidence threshold of {confidence_threshold}, is {round((correct / 17820) * 100, 2)}%")


if __name__ == '__main__':
    test_model()
