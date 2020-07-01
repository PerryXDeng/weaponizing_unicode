import generate_text.generate_character as draw_char
import pickle
import numpy as np
import random
import cv2 as cv

FONTS_PATH = '../fonts/'


def compile_datasets(training_size, test_size, font_size=.2, img_size=400):
    infile = open(FONTS_PATH + 'multifont_mapping.pkl', 'rb')
    unicode_mapping_dict = pickle.load(infile)
    unicode_count = len(unicode_mapping_dict)
    infile.close()
    unicode_chars_available = list(unicode_mapping_dict.keys())
    unicode_chars_population = list(unicode_mapping_dict.keys())
    anchors = np.empty((training_size, img_size, img_size), dtype=np.uint8)
    positives = np.empty((training_size, img_size, img_size), dtype=np.uint8)
    negatives = np.empty((training_size, img_size, img_size), dtype=np.uint8)
    x1_test = np.empty((test_size, img_size, img_size), dtype=np.uint8)
    x2_test = np.empty((test_size, img_size, img_size), dtype=np.uint8)
    y_test = np.arange(test_size) % 2
    for i in range(training_size):
        anchor_char = unicode_chars_available[random.randint(0, len(unicode_chars_available) - 1)]
        unicode_chars_available.remove(anchor_char)
        negative_char = anchor_char
        while negative_char == anchor_char:
            negative_char = unicode_chars_population[random.randint(0, unicode_count - 1)]
        supported_anchor_fonts = unicode_mapping_dict[anchor_char]
        supported_negative_fonts = unicode_mapping_dict[negative_char]
        anchor_font = FONTS_PATH + supported_anchor_fonts[random.randint(0, len(supported_anchor_fonts) - 1)]
        positive_font = FONTS_PATH + supported_anchor_fonts[random.randint(0, len(supported_anchor_fonts) - 1)]
        negative_font = FONTS_PATH + supported_negative_fonts[random.randint(0, len(supported_negative_fonts) - 1)]
        anchors[i] = draw_char.transformImg(draw_char.drawChar(img_size, chr(anchor_char), font_size, anchor_font))
        positives[i] = draw_char.transformImg(draw_char.drawChar(img_size, chr(anchor_char), font_size, positive_font))
        negatives[i] = draw_char.transformImg(
            draw_char.drawChar(img_size, chr(negative_char), font_size, negative_font))
    for i in range(test_size):
        x1_char = unicode_chars_available[random.randint(0, len(unicode_chars_available) - 1)]
        unicode_chars_available.remove(x1_char)
        x2_char = x1_char
        if y_test[i] == 0:
            while x2_char == x1_char:
                x2_char = unicode_chars_population[random.randint(0, unicode_count - 1)]
        supported_x1_fonts = unicode_mapping_dict[x1_char]
        supported_x2_fonts = unicode_mapping_dict[x2_char]
        x1_font = FONTS_PATH + supported_x1_fonts[random.randint(0, len(supported_x1_fonts) - 1)]
        x2_font = FONTS_PATH + supported_x2_fonts[random.randint(0, len(supported_x2_fonts) - 1)]
        x1_test[i] = draw_char.transformImg(draw_char.drawChar(img_size, chr(x1_char), font_size, x1_font))
        x2_test[i] = draw_char.transformImg(draw_char.drawChar(img_size, chr(x2_char), font_size, x2_font))
    return anchors, positives, negatives, x1_test, x2_test, y_test


def test_drawing(unicode_mapping_dict, font_size=.2, img_size=400):
    # 17 corrupt unicode chars
    # 3600 invalid pixel size
    for i in unicode_mapping_dict.keys():
        try:
            a = random.randint(0, len(unicode_mapping_dict[i]) - 1)
            # print(a)
            draw_char.transformImg(
                draw_char.drawChar(img_size, chr(i), font_size, FONTS_PATH + unicode_mapping_dict[i][a]))
        except (ValueError, OSError) as e:
            print(e, i, unicode_mapping_dict[i][a])


def display_chars(unicode_mapping_dict, display_test, display_train):
    anchors, positives, negatives, x1_test, x2_test, y_test = unicode_mapping_dict
    for i in range(display_train):
        cv.imshow('anchor', anchors[i])
        cv.imshow('positive', positives[i])
        cv.imshow('negative', negatives[i])
        cv.waitKey(0)
        cv.destroyAllWindows()
    for i in range(display_test):
        cv.imshow('x1', x1_test[i])
        cv.imshow('x2', x2_test[i])
        print(y_test[i])
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    # Generate the mapping file
    infile = open(FONTS_PATH + 'multifont_mapping.pkl', 'rb')
    unicode_mapping_dict = pickle.load(infile)
    infile.close()

    # display_chars(unicode_mapping_dict, 50,0)
    # test_drawing(unicode_mapping_dict)
