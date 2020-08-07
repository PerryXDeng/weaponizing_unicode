import argparse
import tensorflow as tf
import pickle
import os
from generate_datasets import try_draw_single_font
from train_triplet_loss_modular import floatify_and_normalize
import numpy as np
import cv2 as cv

def cos_distance_sum(x1, x2):
  x1_axis_1 = x1.shape[0]
  x1_axis_2 = x1.shape[1]
  x2_axis_1 = x2.shape[0]
  x2_axis_2 = x2.shape[1]
  a_v = tf.reshape(x1, [x1_axis_1, 1, x1_axis_2])
  b_v = tf.reshape(x2, [x2_axis_1, x2_axis_2, 1])
  cos_sim_vect = tf.reshape(tf.matmul(a_v, b_v), [x1_axis_1]) / ((tf.norm(x1, axis=1) * tf.norm(x2, axis=1)) + 1e-5)
  return tf.reduce_sum(tf.abs(cos_sim_vect)).numpy()

# Keys are fonts
def find_median_vector(font_features_dict):
    if len(font_features_dict) < 3:
        return next(iter(font_features_dict.values())), 99999
    median_vector_cos_dist_sum = 10e5
    median_vector_font = ""
    for font in font_features_dict.keys():
        temp = font_features_dict.copy()
        temp.pop(font)
        feature_vector_tensor = tf.reshape(tf.convert_to_tensor(font_features_dict[font]), [1, -1])
        other_vectors_tensor = tf.convert_to_tensor(list(temp.values()))
        vector_cos_dist_sum = cos_distance_sum(other_vectors_tensor, feature_vector_tensor)
        if vector_cos_dist_sum < median_vector_cos_dist_sum:
            median_vector_cos_dist_sum = vector_cos_dist_sum
            median_vector_font = font
    return font_features_dict[median_vector_font], median_vector_cos_dist_sum / len(font_features_dict)



def _load_model_load_data_and_extract_features(model_path: str, batch_size: int, multifont_mapping_path: str) -> dict:
    # Load model + weights
    json_file = open(os.path.join(model_path, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(os.path.join(model_path, 'model.h5'))

    # Load model input info
    model_info_file = open(os.path.join(model_path, 'model_info.pkl'), 'rb')
    model_info_dict = pickle.load(model_info_file)
    img_size, font_size = model_info_dict['img_size'], model_info_dict['font_size']
    empty_image = np.full((img_size, img_size), 255)

    # Load multifont mapping dict
    multifont_mapping_file = open(multifont_mapping_path, 'rb')
    unicode_mapping_dict = pickle.load(multifont_mapping_file)

    unicode_features_dict = {}
    unicode_batch = {}
    for unicode_point in unicode_mapping_dict.keys():
        print(unicode_point)
        for font_path in unicode_mapping_dict[unicode_point]:
            drawn_unicode = try_draw_single_font(unicode_point, font_path, empty_image, img_size, font_size, "./fonts",
                                                 transform_img=False)
            if not (drawn_unicode == empty_image).all():
                drawn_unicode_preprocessed = floatify_and_normalize(cv.cvtColor(drawn_unicode, cv.COLOR_GRAY2RGB))
                unicode_batch[str(unicode_point) + "|" + font_path] = drawn_unicode_preprocessed
            if len(unicode_batch) == batch_size:
                # The predict function outputs a numpy array on the CPU, not a tensor on the GPU!
                batch_forward = loaded_model.predict(tf.convert_to_tensor(list(unicode_batch.values())))
                unicode_batch = list(zip(unicode_batch.keys(), batch_forward))
                for unicode_font, feature_vector in unicode_batch:
                    feature_vector_font_code = unicode_font.split('|')
                    if feature_vector_font_code[0] not in unicode_features_dict:
                        unicode_features_dict[feature_vector_font_code[0]] = {}
                    unicode_features_dict[feature_vector_font_code[0]][feature_vector_font_code[1]] = feature_vector
                unicode_batch = {}
    for unicode_point in unicode_features_dict.keys():
        print("STARTING THIS FONT")
        unicode_features_dict[unicode_point] = find_median_vector(unicode_features_dict[unicode_point])
        print(unicode_features_dict[unicode_point].shape)
        print("ENDING THIS FONT")
    return unicode_features_dict


def font_dict_to_median_dict(codepoints_features_map: str):
    codepoints_features_map_file = open(codepoints_features_map, 'rb')
    codepoints_features_map_dict = pickle.load(codepoints_features_map_file)
    median_feature_vect_dict = {}
    for unicode_point in codepoints_features_map_dict.keys():
        font_dict = codepoints_features_map_dict[unicode_point]
        median_feature_vect, min_value = find_median_vector(font_dict)
        median_feature_vect_dict[unicode_point] = median_feature_vect
        print(unicode_point, min_value, len(font_dict))
    with open("codepoints_median_features_map.pkl", 'wb+') as f:
        pickle.dump(median_feature_vect_dict, f)


if __name__ == '__main__':
    # WOW = _load_model_load_data_and_extract_features(model_path="./model_1", batch_size=45,
    #                                                  multifont_mapping_path="./fonts/multifont_mapping.pkl")
    # print(len(WOW))
    # with open("codepoints_features_map.pkl", 'wb+') as f:
    #     pickle.dump(WOW, f)
    font_dict_to_median_dict("./codepoints_features_map.pkl")
