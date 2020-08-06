import requests
import pickle
import numpy as np
import tensorflow as tf
import random
from generate_datasets import try_draw_char_all_fonts


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
    return consortium_clusters_dict, 4248


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
    consortium_clusters_dict, total_puny_pairs = get_consortium_clusters_dict()
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
    consortium_clusters_dict, total_puny_pairs = get_consortium_clusters_dict()
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
    consortium_clusters_dict, total_puny_pairs = get_consortium_clusters_dict()
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


def mergeSort_most_supported_font(font_unicodes_list, remove_list):
    if len(font_unicodes_list) == 1:
        ye = [(font_unicodes_list[0][0], [i for i in font_unicodes_list[0][1] if i not in remove_list])]
        return ye
    else:
        mid = len(font_unicodes_list) // 2
        a = mergeSort_most_supported_font(font_unicodes_list[:mid], remove_list)
        b = mergeSort_most_supported_font(font_unicodes_list[mid:], remove_list)
        c = []
        a_count = 0
        b_count = 0
        while a_count < len(a) or b_count < len(b):
            if a_count == len(a):
                c.append(b[b_count])
                b_count += 1
            elif b_count == len(b):
                c.append(a[a_count])
                a_count += 1
            else:
                if len(a[a_count][1]) > len(b[b_count][1]):
                    c.append(a[a_count])
                    a_count += 1
                else:
                    c.append(b[b_count])
                    b_count += 1
        return c


def get_minimum_font_dict_merge(font_unicodes_list, target_characters):
    a = {}
    copied_font_unicodes_list = font_unicodes_list.copy()
    while len(a) < target_characters:
        copied_font_unicodes_list = mergeSort_most_supported_font(copied_font_unicodes_list, list(a.values()))
        for i in range(3):
            print(copied_font_unicodes_list[i][0], len(copied_font_unicodes_list[i][1]))
        most_supported_font = copied_font_unicodes_list[0]
        for unicode in most_supported_font[1]:
            assert unicode not in a
            a[unicode] = most_supported_font[0]
        del copied_font_unicodes_list[0]
    return a


def get_minimum_font_dict(font_unicodes_list, target_characters):
    a = {}
    last_update = []
    copied_font_unicodes_list = font_unicodes_list.copy()
    while len(a) < target_characters:
        max_len_index = -1
        max_len = 0
        for char_ in range(len(copied_font_unicodes_list)):
            copied_font_unicodes_list[char_] = (copied_font_unicodes_list[char_][0],
                                                [i for i in copied_font_unicodes_list[char_][1] if
                                                 i not in last_update])
            font_len = len(copied_font_unicodes_list[char_][1])
            if font_len > max_len:
                max_len = font_len
                max_len_index = char_
        most_supported_font = copied_font_unicodes_list[max_len_index]
        for unicode in most_supported_font[1]:
            assert unicode not in a
            a[unicode] = most_supported_font[0]
        last_update = most_supported_font[1]
        print(most_supported_font[0], max_len)
        del copied_font_unicodes_list[max_len_index]
    return a


# Complexity: Where F is # of fonts & N is # of characters
def create_minimum_font_dict(unicode_all_feature_vectors_dict_path):
    unicode_all_feature_vectors_dict = pickle.load(open(unicode_all_feature_vectors_dict_path, 'rb'))
    target_num_unicode_char = len(unicode_all_feature_vectors_dict)
    minimum_font_dict = {}
    for unicode, font_feature_dict in unicode_all_feature_vectors_dict.items():
        supported_fonts = font_feature_dict.keys()
        for font in supported_fonts:
            if font not in minimum_font_dict:
                minimum_font_dict[font] = [unicode]
            else:
                minimum_font_dict[font].append(unicode)
    return get_minimum_font_dict(list(minimum_font_dict.items()), target_num_unicode_char)


def get_consortium_clusters_model_accuracy_min_used_fonts(unicode_all_feature_vectors_dict_path, font_dict_path,
                                                          cos_threshold):
    consortium_clusters_dict, total_puny_pairs = get_consortium_clusters_dict()
    unicode_supported_fonts_dict = pickle.load(open(unicode_all_feature_vectors_dict_path, 'rb'))
    unicode_font_dict = pickle.load(open(font_dict_path, 'rb'))
    total_supported_puny_pairs = 0
    total_correct = 0
    for unicode_target in consortium_clusters_dict.keys():
        if unicode_target in unicode_supported_fonts_dict:
            for unicode_source in consortium_clusters_dict[unicode_target]:
                if unicode_source in unicode_supported_fonts_dict:
                    total_supported_puny_pairs += 1
                    target_vect = convert_to_tensor_and_reshape(
                        unicode_supported_fonts_dict[unicode_target][unicode_font_dict[unicode_target]])
                    source_vect = convert_to_tensor_and_reshape(
                        unicode_supported_fonts_dict[unicode_source][unicode_font_dict[unicode_source]])
                    cos_sim_ = cos_distance(target_vect, source_vect)
                    if cos_sim_ > cos_threshold:
                        total_correct += 1
    print(total_correct / total_supported_puny_pairs)


def generate_consortium_feature_vects_min_used_fonts(unicode_all_feature_vectors_dict_path, font_dict_path,
                                                     n_clusters):
    consortium_clusters_dict, total_puny_pairs = get_consortium_clusters_dict()
    unicode_supported_fonts_dict = pickle.load(open(unicode_all_feature_vectors_dict_path, 'rb'))
    unicode_font_dict = pickle.load(open(font_dict_path, 'rb'))
    consortium_feature_vects = {}
    print(len(consortium_clusters_dict))
    for unicode_target in list(consortium_clusters_dict.keys())[:n_clusters]:
        if unicode_target in unicode_supported_fonts_dict:
            for unicode_source in consortium_clusters_dict[unicode_target]:
                if unicode_source in unicode_supported_fonts_dict:
                    if unicode_target not in consortium_feature_vects:
                        consortium_feature_vects[unicode_target] = unicode_supported_fonts_dict[unicode_target][
                            unicode_font_dict[unicode_target]]
                    if unicode_source not in consortium_feature_vects:
                        consortium_feature_vects[unicode_source] = unicode_supported_fonts_dict[unicode_source][
                            unicode_font_dict[unicode_source]]
    with open(f"consortium_feature_vects_{n_clusters}_clusters.pkl", 'wb+') as f:
        pickle.dump(consortium_feature_vects, f)
    print(len(consortium_feature_vects))


if __name__ == '__main__':
    # get_consortium_clusters_model_accuracy_median_vector("./codepoints_features_map.pkl",.4)
    # get_consortium_clusters_model_accuracy_random_font("./codepoints_features_map_supported_fonts.pkl",.4)
    # ye_dict = create_minimum_font_dict("./codepoints_features_map_supported_fonts.pkl")
    # with open("min_supported_fonts.pkl", 'wb+') as f:
    #     pickle.dump(ye_dict, f)
    # get_consortium_clusters_model_accuracy_min_used_fonts("./codepoints_features_map_supported_fonts.pkl",
    #                                                       "min_supported_fonts.pkl", .4)
    # get_consortium_clusters_model_accuracy_median_vector("./codepoints_features_map.pkl", .4)
    
    #generate_consortium_feature_vects_min_used_fonts("./codepoints_features_map_supported_fonts.pkl",
                                                     #"min_supported_fonts.pkl", 250)
    unicode_supported_fonts_dict = pickle.load(open("codepoints_cluster_map.pkl", 'rb'))
    for a,b in unicode_supported_fonts_dict.items():
      print(a)
      print(b)
