import requests
import pickle
import numpy as np
import tensorflow as tf
import random
from feature_cluster_algos import CosineSimGraphClustererGPU


def get_consortium_clusters_dict():
    url = 'https://www.unicode.org/Public/security/12.0.0/confusables.txt'
    # Load Text File
    r = requests.get(url, allow_redirects=True)
    punycodes_list = r.text[385:].split("\n")
    punycodes_list = punycodes_list[:len(punycodes_list) - 3]
    # 4248 pairs
    # 3353 we support :(
    consortium_clusters_dict = {}
    for i in punycodes_list:
        if i is not '':
            punycode_pair = i.split(";")[:2]
            if len(punycode_pair[1]) < 8:
                source = str(int(punycode_pair[0].replace(' ', ''), 16))
                target = str(int(punycode_pair[1].replace(' ', '').replace('\t', ''), 16))
                if target not in consortium_clusters_dict:
                    consortium_clusters_dict[target] = [source]
                else:
                    consortium_clusters_dict[target].append(source)
    return consortium_clusters_dict


def cos_distance(x1, x2):
    x1_axis_1 = x1.shape[0]
    x1_axis_2 = x1.shape[1]
    x2_axis_1 = x2.shape[0]
    x2_axis_2 = x2.shape[1]
    a_v = tf.reshape(x1, [x1_axis_1, 1, x1_axis_2])
    b_v = tf.reshape(x2, [x2_axis_1, x2_axis_2, 1])
    cos_sim_vect = tf.reshape(tf.matmul(a_v, b_v), [x1_axis_1]) / ((tf.norm(x1, axis=1) * tf.norm(x2, axis=1)) + 1e-5)
    return cos_sim_vect.numpy()


def get_best_cos_sim(dict_1, dict_2):
    best_sim_ = -1
    selected_feature_vects = list(dict_1.values())
    target_feature_vects = tf.convert_to_tensor(list(dict_2.values()))
    for feature_vect in selected_feature_vects:
        feature_vect_reshaped = convert_to_tensor_and_reshape(feature_vect)
        sim = cos_distance(target_feature_vects, feature_vect_reshaped)
        assert sim.shape() == len(dict_2)
        max_sim = np.max(sim)
        if max_sim > best_sim_:
            best_sim_ = max_sim
    assert best_sim_ != -1
    return best_sim_


def convert_to_tensor_and_reshape(numpy_array):
    return tf.reshape(tf.convert_to_tensor(numpy_array), [1, -1])


def get_consortium_clusters_model_accuracy_median_vector(unicode_median_feature_vector_dict_path, cos_threshold):
    consortium_clusters_dict = get_consortium_clusters_dict()
    unicode_median_feature_vector_dict = pickle.load(open(unicode_median_feature_vector_dict_path, 'rb'))
    total_supported_puny_pairs = 0
    total_correct = 0
    for unicode_target in consortium_clusters_dict.keys():
        if unicode_target in unicode_median_feature_vector_dict:
            for unicode_source in consortium_clusters_dict[unicode_target]:
                if unicode_source in unicode_median_feature_vector_dict:
                    total_supported_puny_pairs += 1
                    sim_ = cos_distance(
                        convert_to_tensor_and_reshape(unicode_median_feature_vector_dict[unicode_target]),
                        convert_to_tensor_and_reshape(unicode_median_feature_vector_dict[unicode_source]))
                    if sim_ > cos_threshold:
                        total_correct += 1
    print(total_correct / total_supported_puny_pairs)


def get_consortium_clusters_model_accuracy_best_font(unicode_all_feature_vectors_dict_path, cos_threshold):
    consortium_clusters_dict = get_consortium_clusters_dict()
    unicode_median_feature_vector_dict = pickle.load(open(unicode_all_feature_vectors_dict_path, 'rb'))
    total_supported_puny_pairs = 0
    total_correct = 0
    for unicode_target in consortium_clusters_dict.keys():
        if unicode_target in unicode_median_feature_vector_dict:
            for unicode_source in consortium_clusters_dict[unicode_target]:
                if unicode_source in unicode_median_feature_vector_dict:
                    total_supported_puny_pairs += 1
                    best_sim_ = get_best_cos_sim(unicode_median_feature_vector_dict[unicode_target],
                                                 unicode_median_feature_vector_dict[unicode_source])
                    if best_sim_ > cos_threshold:
                        total_correct += 1
    print(total_correct / total_supported_puny_pairs)


def get_consortium_clusters_model_accuracy_random_font(unicode_all_feature_vectors_dict_path, cos_threshold):
    consortium_clusters_dict = get_consortium_clusters_dict()
    unicode_all_feature_vectors_dict = pickle.load(open(unicode_all_feature_vectors_dict_path, 'rb'))
    total_supported_puny_pairs = 0
    total_correct = 0
    for unicode_target in consortium_clusters_dict.keys():
        if unicode_target in unicode_all_feature_vectors_dict:
            for unicode_source in consortium_clusters_dict[unicode_target]:
                if unicode_source in unicode_all_feature_vectors_dict:
                    total_supported_puny_pairs += 1
                    target_features = list(unicode_all_feature_vectors_dict[unicode_target].values())
                    source_features = list(unicode_all_feature_vectors_dict[unicode_source].values())
                    target_random_feature = target_features[random.randint(0, len(target_features) - 1)]
                    source_random_feature = source_features[random.randint(0, len(source_features) - 1)]
                    target_features_preprocessed = convert_to_tensor_and_reshape(target_random_feature)
                    source_features_preprocessed = convert_to_tensor_and_reshape(source_random_feature)
                    sim_ = cos_distance(target_features_preprocessed, source_features_preprocessed)
                    if sim_ > cos_threshold:
                        total_correct += 1
    print(total_correct / total_supported_puny_pairs)


def calculate_consortium_cluster_accuracy(features_dict_file_path, cos_threshold):
    consortium_clusters_dict = get_consortium_clusters_dict()
    features_dict_file = pickle.load(open(features_dict_file_path, 'rb'))
    total_supported_puny_pairs = 0
    total_correct = 0
    for unicode_target in consortium_clusters_dict.keys():
        if unicode_target in features_dict_file:
            for unicode_source in consortium_clusters_dict[unicode_target]:
                if unicode_source in features_dict_file:
                    total_supported_puny_pairs += 1
                    target_vect = convert_to_tensor_and_reshape(
                        features_dict_file[unicode_target])
                    source_vect = convert_to_tensor_and_reshape(
                        features_dict_file[unicode_source])
                    cos_sim_ = cos_distance(target_vect, source_vect)
                    if cos_sim_ > cos_threshold:
                        total_correct += 1
    print(total_correct / total_supported_puny_pairs)


def generate_consortium_clusters(n_clusters):
    consortium_clusters_dict = get_consortium_clusters_dict()
    cos_Clusterer = CosineSimGraphClustererGPU(save_dir="./", threshold=.7, epsilon=1e-5)
    codepoints_cluster_map, cluster_codepoints_map = cos_Clusterer._cluster_features_into_equivalence_classes(
        consortium_clusters_dict[:n_clusters])


if __name__ == '__main__':
    generate_consortium_clusters(100)