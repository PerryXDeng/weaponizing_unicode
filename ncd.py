# see first article in https://www.springerprofessional.de/en/web-information-system-engineering-wise-2011/3755202?tocPage=1
# for context, ncd stands for normalized compression distance

import lzma
import numpy as np
import random
import pickle
import os

from sklearn import svm
from sklearn.metrics import average_precision_score, precision_recall_curve
from matplotlib import pyplot as plt

from unicode_info.database import generate_data_for_experiment, generate_positive_pairs_consortium, generate_negative_pairs_consortium
from generate_datasets import try_draw_single_font


def C(x: bytes):
    """
    gives the compressed length of a byte string
    """
    return len(lzma.compress(x))


def ncd(x: bytes, y: bytes):
    """
    calculates the normalized compression distance between two bytes string
    """
    Cx = C(x)
    Cy = C(y)
    Cxy = C(x + y)
    return (Cxy - min(Cx, Cy)) / max(Cx, Cy)


def ncd_ndarray(x: np.ndarray, y: np.ndarray):
    return ncd(x.tobytes(), y.tobytes())


def train_svm_generate_statistics_and_auc(similarities: np.ndarray, compression_dists: np.ndarray, labels: np.ndarray):
    """
    dim measures = [n], dtype = float
    dim labels = [n], dtype = int
    """
    classifier = svm.SVC(kernel='linear')

    measures = similarities.reshape(-1, 1)# dim [n, 1]
    classifier.fit(measures, labels)
    y_score = classifier.decision_function(measures)
    deep_acc = classifier.score(measures, labels)
    deep_ap = average_precision_score(labels, y_score)
    deep_precision, deep_recall, _ = precision_recall_curve(labels, y_score)

    measures = compression_dists.reshape(-1, 1)# dim [n, 1]
    classifier.fit(measures, labels)
    y_score = classifier.decision_function(measures)
    ncd_acc = classifier.score(measures, labels)
    ncd_ap = average_precision_score(labels, y_score)
    ncd_precision, ncd_recall, _ = precision_recall_curve(labels, y_score)

    plt.figure()
    plt.rcParams.update({'font.size': 22})
    lines = []
    labs = []
    lines.append(plt.plot(deep_recall, deep_precision, color="C0", lw=2)[0])
    labs.append("Our Approach, Acc.: {0:0.3f}, AP: {1:0.3f}".format(deep_acc, deep_ap))
    lines.append(plt.plot(ncd_recall, ncd_precision, color="C1", lw=2)[0])
    labs.append("NCD Approach, Acc.: {0:0.3f}, AP: {1:0.3f}".format(ncd_acc, ncd_ap))
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(lines, labs, loc='lower left', prop=dict(size=22))
    plt.show()


def comparison():
    num_pairs = 1000

    supported_consortium_feature_vectors, supported_consortium_clusters_dict = generate_data_for_experiment()

    min_supported_fonts_dict = pickle.load(open('min_supported_fonts.pkl', 'rb'))

    # Load model input info
    model_info_file = open(os.path.join('model_1', 'model_info.pkl'), 'rb')
    model_info_dict = pickle.load(model_info_file)
    img_size, font_size = model_info_dict['img_size'], model_info_dict['font_size']
    empty_image = np.full((img_size, img_size), 255)

    positive_pairs = generate_positive_pairs_consortium(supported_consortium_clusters_dict, num_pairs)
    negative_pairs = generate_negative_pairs_consortium(supported_consortium_clusters_dict, num_pairs)
    pairs = positive_pairs + negative_pairs

    labels = np.zeros(num_pairs * 2, dtype=int)
    labels[0:num_pairs] = 1

    cosine_similarities = np.empty(num_pairs * 2, dtype=float)
    normalized_compression_distances = np.empty(num_pairs * 2, dtype=float)

    for i in range(num_pairs * 2):
        code_x, code_y = pairs[i]
        features_x = supported_consortium_feature_vectors[code_x]
        features_y = supported_consortium_feature_vectors[code_y]
        glyph_x = try_draw_single_font(int(code_x), min_supported_fonts_dict[code_x], empty_image, img_size,
                                                font_size, "./fonts", transform_img=False)
        glyph_y = try_draw_single_font(int(code_y), min_supported_fonts_dict[code_y], empty_image, img_size,
                                                font_size, "./fonts", transform_img=False)
        cosine_similarities[i] = np.dot(features_x, features_y) / (
                np.linalg.norm(features_x) * np.linalg.norm(features_y))
        normalized_compression_distances[i] = ncd_ndarray(glyph_x, glyph_y)

    train_svm_generate_statistics_and_auc(cosine_similarities, normalized_compression_distances, labels)

if __name__ == '__main__':
    comparison()
