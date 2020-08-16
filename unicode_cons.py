import requests
import pickle
import numpy as np
import cupy as cp
import tensorflow as tf
import random

from feature_cluster_algos import CosineSimGraphClustererCPU
from cluster_metrics import calculate_mean_iou, calculate_mean_coverage


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


def generate_supported_consortium_feature_vectors_and_clusters_dict(n_clusters, features_dict_file_path):
    consortium_clusters_dict = get_consortium_clusters_dict()
    # print(len(consortium_clusters_dict))
    features_dict = pickle.load(open(features_dict_file_path, 'rb'))

    supported_consortium_feature_vectors = {}
    supported_consortium_clusters_dict = {}
    for cluster_source in consortium_clusters_dict.keys():
        if cluster_source in features_dict:
            supported_consortium_clusters_dict[cluster_source] = []
            for target in consortium_clusters_dict[cluster_source]:
                if target in features_dict:
                    supported_consortium_clusters_dict[cluster_source].append(target)
                    if cluster_source not in supported_consortium_feature_vectors:
                        supported_consortium_feature_vectors[cluster_source] = features_dict[cluster_source]
                    if target not in supported_consortium_feature_vectors:
                        supported_consortium_feature_vectors[target] = features_dict[target]
            if len(supported_consortium_clusters_dict[cluster_source]) == 0:
                del supported_consortium_clusters_dict[cluster_source]
        if len(supported_consortium_clusters_dict) == n_clusters:
            break
    return supported_consortium_feature_vectors, supported_consortium_clusters_dict


def generate_suppported_consortium_clusters(n_clusters, features_dict_file_path, cos_threshold):
    supported_consortium_feature_vectors, supported_consortium_clusters_dict = generate_supported_consortium_feature_vectors_and_clusters_dict(
        n_clusters, features_dict_file_path)
    # print(len(supported_consortium_feature_vectors))
    # print(len(supported_consortium_clusters_dict))
    cos_Clusterer = CosineSimGraphClustererCPU(save_dir="./", threshold=cos_threshold, epsilon=1e-5)
    codepoints_cluster_map, cluster_codepoints_map = cos_Clusterer._cluster_features_into_equivalence_classes(
        supported_consortium_feature_vectors)
    return codepoints_cluster_map, cluster_codepoints_map, supported_consortium_clusters_dict


def convert(dict_):
    dict_copy = dict_.copy()
    returner = {}
    count = 0
    for key, value in dict_copy.items():
        value.append(key)
        returner[count] = value
        count += 1
    return returner


def _generate_adjacency_matrix(features):
    # gpu [n, k]
    ordered_features_gpu = cp.array(features)
    n, k = ordered_features_gpu.shape

    a = ordered_features_gpu.reshape((n, 1, 1, k))
    b = ordered_features_gpu.reshape((1, n, k, 1))
    # [n, n]
    dot_products = cp.matmul(a, b).reshape((n, n))

    # [n]
    norms = cp.linalg.norm(ordered_features_gpu, axis=1)

    norms_a = norms.reshape((n, 1))
    norms_b = norms.reshape((1, n))  # same as the above but transposed
    # [n, n]
    norms_prod = cp.multiply(norms_a, norms_b)
    cosine_similarity = dot_products / (norms_prod + 1e-7)
    return cp.asnumpy(cosine_similarity)


def _generate_statistics(converted_dict, features_dict_file_path):
    features_dict = pickle.load(open(features_dict_file_path, 'rb'))
    mean = 0
    count = 0
    std_dev = 0
    for cluster in converted_dict.values():
        ordered_features = np.empty([len(cluster), len(features_dict[cluster[0]])], dtype=np.float32)
        for i in range(len(cluster)):
            ordered_features[i] = features_dict[cluster[i]]
        if len(cluster) > 2:
            cos_matrix = _generate_adjacency_matrix(ordered_features)
            stats = np.tril(cos_matrix, -1)
            stats = stats[stats != 0]
            mean += np.mean(stats)
            std_dev += np.std(stats)
            count += 1
    print(mean / count)
    print(std_dev / count)


def normalize_rows(vector):
    return vector / (np.linalg.norm(vector, axis=1).reshape((vector.shape[0], 1)) + 1e-8)


def calculate_centroid(feature_vectors):
    normalized_vectors = normalize_rows(feature_vectors)
    centroid_ = np.sum(normalized_vectors, axis=0)
    return normalize_rows(centroid_.reshape((1, -1)))


def convert_to_clusters_codepoints_map(dict_):
    a = {}
    for key, value in dict_.items():
        a[key] = list(value.keys())
    return a


def convert_to_codepoints_clusters_map(dict_):
    returner = {}
    for key, values in dict_.items():
        for unicode in values:
            returner[unicode] = key
    return returner


def cos_sim_matrix(matrix_a, matrix_b):
    cos_sim = []
    for x_ in matrix_a:
        for y_ in matrix_b:
            cos_sim.append(np.matmul(x_, y_) / (np.linalg.norm(x_) * np.linalg.norm(y_)))
    return cos_sim


def combine_clusters(clustered_initial, target_thresh):
    clustered = clustered_initial.copy()
    cluster_key_list = list(clustered.keys())
    i = 0
    while i < len(cluster_key_list) - 2:
        j = i + 1
        deleted = []
        while j < len(cluster_key_list) - 1:
            feature_vector_a = np.stack(list(clustered[cluster_key_list[i]].values()))
            feature_vector_b = np.stack(list(clustered[cluster_key_list[j]].values()))
            cosine_similarity = cos_sim_matrix(feature_vector_a,
                                               feature_vector_b)
            if cosine_similarity > target_thresh.all():
                deleted.append(cluster_key_list[j])
                clustered[cluster_key_list[i]].update(clustered[cluster_key_list[j]])
            j += 1
        for d in deleted:
            del clustered[d]
            cluster_key_list.remove(d)
        i += 1
    return clustered


def combine_clusters_adj(clustered_initial, target_mean, target_std_dev):
    clustered = clustered_initial.copy()
    cluster_key_list = list(clustered.keys())
    i = 0
    while i < len(cluster_key_list) - 2:
        j = i + 1
        deleted = []
        while j < len(cluster_key_list) - 1:
            feature_vector_a = np.stack(list(clustered[cluster_key_list[i]].values()))
            feature_vector_b = np.stack(list(clustered[cluster_key_list[j]].values()))
            stacked_features = np.concatenate((feature_vector_a, feature_vector_b))
            cos_matrix = _generate_adjacency_matrix(stacked_features)
            cosine_similarity = np.tril(cos_matrix, -1)
            cosine_similarity = cosine_similarity[cosine_similarity != 0]
            proposal_mean = np.mean(cosine_similarity)
            proposal_std = np.std(cosine_similarity)
            if proposal_mean > target_mean and proposal_std < target_std_dev:
                deleted.append(cluster_key_list[j])
                clustered[cluster_key_list[i]].update(clustered[cluster_key_list[j]])
            j += 1
        for d in deleted:
            del clustered[d]
            cluster_key_list.remove(d)
        i += 1
    return clustered


def cluster_test(n_clusters, m, s, t):
    supported_consortium_feature_vectors, supported_consortium_clusters_dict = generate_supported_consortium_feature_vectors_and_clusters_dict(
        n_clusters, 'features_dict_file.pkl')
    ground_truth_consoritium_codepoints_map = convert(supported_consortium_clusters_dict)
    clustered = {}
    for unicode_, feature_vect in supported_consortium_feature_vectors.items():
        add = 0
        for cluster_key, cluster_dict_code_feature_vector in clustered.items():
            cluster_features = np.stack(list(cluster_dict_code_feature_vector.values()))
            feature_vect_reshaped = feature_vect.reshape((1, -1))
            cosine_similarity = cos_distance(cluster_features, feature_vect_reshaped)
            adjacency_matrix = np.asarray(cosine_similarity > t)
            if (adjacency_matrix == True).all():
                clustered[cluster_key][unicode_] = feature_vect
                add = 1
                break
        if add == 0:
            clustered[len(clustered)] = {unicode_: feature_vect}

    clustered = combine_clusters_adj(clustered, m, s)
    print(m, s, t, generate_mean_iou(clustered, ground_truth_consoritium_codepoints_map))
    print()


def generate_mean_iou(cluster_predictions, ground_truth_consoritium_codepoints_map):
    cluster_predictions_codepoints_map = convert_to_clusters_codepoints_map(cluster_predictions)
    cluster_predictions_clusters_map = convert_to_codepoints_clusters_map(cluster_predictions_codepoints_map)
    return calculate_mean_iou(cluster_predictions_clusters_map, cluster_predictions_codepoints_map,
                              ground_truth_consoritium_codepoints_map)


def cluster_test_with_random_characters(n_clusters, m, s, t, target_random_characters):
    supported_consortium_feature_vectors, supported_consortium_clusters_dict = generate_supported_consortium_feature_vectors_and_clusters_dict(
        n_clusters, 'features_dict_file.pkl')
    clusters_unicode_characters = list(supported_consortium_feature_vectors.keys())

    unicode_median_feature_vector_dict = pickle.load(open('features_dict_file.pkl', 'rb'))
    all_unicode_characters = list(unicode_median_feature_vector_dict.keys())

    added_random_characters = 0
    while added_random_characters < target_random_characters:
        possible_random_character = all_unicode_characters.pop(random.randint(0, len(all_unicode_characters) - 1))
        if possible_random_character not in clusters_unicode_characters:
            supported_consortium_feature_vectors[possible_random_character] = unicode_median_feature_vector_dict[
                possible_random_character]
            added_random_characters += 1

    ground_truth_consoritium_codepoints_map = convert(supported_consortium_clusters_dict)
    clustered = {}
    for unicode_, feature_vect in supported_consortium_feature_vectors.items():
        add = 0
        for cluster_key, cluster_dict_code_feature_vector in clustered.items():
            cluster_features = np.stack(list(cluster_dict_code_feature_vector.values()))
            feature_vect_reshaped = feature_vect.reshape((1, -1))
            cosine_similarity = cos_distance(cluster_features, feature_vect_reshaped)
            adjacency_matrix = np.asarray(cosine_similarity > t)
            if (adjacency_matrix == True).all():
                clustered[cluster_key][unicode_] = feature_vect
                add = 1
                break
        if add == 0:
            clustered[len(clustered)] = {unicode_: feature_vect}

    clustered = combine_clusters_adj(clustered, m, s)
    print(target_random_characters, generate_mean_iou(clustered, ground_truth_consoritium_codepoints_map))
    print()


def run():
    cluster_test(100000, .72, .01, .94)
    cluster_test_with_random_characters(100000, .72, .01, .94, 1000)


if __name__ == '__main__':
    run()

# 0.767601996660232
# 0.0935437240793059
# .4311, .435, .44
# .410 .75, .02
