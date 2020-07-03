import generate_text.generate_character as draw_char
import pickle
import numpy as np
import random
import cv2 as cv

FONTS_PATH = '../fonts/'


def try_draw_char(char, available_fonts, empty_image, img_size, font_size):
    if len(available_fonts) == 0:
        # No fonts support drawing this unicode character, or the unicode character is corrupt!
        return empty_image
    else:
        selected_font = available_fonts[random.randint(0, len(available_fonts) - 1)]
        try:
            img = draw_char.transformImg(draw_char.drawChar(img_size, chr(char), font_size, FONTS_PATH + selected_font))
        except ValueError:
            # The drawn character is to big for the desired region!
            available_fonts.remove(selected_font)
            return try_draw_char(char, available_fonts, empty_image, img_size, font_size)
        if (img == empty_image).all():
            # The selected font does not support drawing this character!
            available_fonts.remove(selected_font)
            return try_draw_char(char, available_fonts, empty_image, img_size, font_size)
        else:
            return img


def compile_datasets(training_size, test_size, font_size=.2, img_size=200, color_format='gray'):
    empty_image = np.full((img_size, img_size), 255)
    infile = open(FONTS_PATH + 'multifont_mapping.pkl', 'rb')
    unicode_mapping_dict = pickle.load(infile)
    print(len(unicode_mapping_dict[63609]))
    print(len(unicode_mapping_dict[63656]))
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
        print(i)
        anchors[i] = empty_image
        positives[i] = empty_image
        negatives[i] = empty_image
        while (anchors[i] == empty_image).all():
            anchor_char = unicode_chars_available[random.randint(0, len(unicode_chars_available) - 1)]
            unicode_chars_available.remove(anchor_char)
            supported_anchor_fonts = unicode_mapping_dict[anchor_char]
            anchors[i] = try_draw_char(anchor_char, supported_anchor_fonts, empty_image, img_size, font_size)
        while (negatives[i] == empty_image).all():
            negative_char = anchor_char
            while negative_char == anchor_char:
                negative_char = unicode_chars_population[random.randint(0, unicode_count - 1)]
            supported_negative_fonts = unicode_mapping_dict[negative_char]
            negatives[i] = try_draw_char(negative_char, supported_negative_fonts, empty_image, img_size, font_size)
        while (positives[i] == empty_image).all():
            # Possible fonts need to be regenerated because the drawing function is bugged
            supported_positive_fonts = unicode_mapping_dict[anchor_char]
            print(anchor_char, len(supported_positive_fonts))
            positives[i] = try_draw_char(anchor_char, supported_positive_fonts, empty_image, img_size, font_size)
    for i in range(test_size):
        print(i)
        x1_test[i] = empty_image
        x2_test[i] = empty_image
        while (x1_test[i] == empty_image).all():
            x1_char = unicode_chars_available[random.randint(0, len(unicode_chars_available) - 1)]
            unicode_chars_available.remove(x1_char)
            supported_x1_fonts = unicode_mapping_dict[x1_char]
            x1_test[i] = try_draw_char(x1_char, supported_x1_fonts, empty_image, img_size, font_size)
        if y_test[i] == 1:
            while (x2_test[i] == empty_image).all():
                # Possible fonts need to be regenerated because the drawing function is bugged
                supported_x2_fonts = unicode_mapping_dict[x1_char]
                x2_test[i] = try_draw_char(x1_char, supported_x2_fonts, empty_image, img_size, font_size)
        else:
            while (x2_test[i] == empty_image).all():
                x2_char = x1_char
                while x2_char == x1_char:
                    x2_char = unicode_chars_population[random.randint(0, unicode_count - 1)]
                supported_x2_fonts = unicode_mapping_dict[x2_char]
                x2_test[i] = try_draw_char(x2_char, supported_x2_fonts, empty_image, img_size, font_size)
    return anchors, positives, negatives, x1_test, x2_test, y_test


def test_drawing(font_size=.2, img_size=400):
    # 17 corrupt unicode chars
    # 3600 invalid pixel size
    infile = open(FONTS_PATH + 'multifont_mapping.pkl', 'rb')
    unicode_mapping_dict = pickle.load(infile)
    infile.close()
    for i in unicode_mapping_dict.keys():
        try:
            a = random.randint(0, len(unicode_mapping_dict[i]) - 1)
            # print(a)
            draw_char.transformImg(
                draw_char.drawChar(img_size, chr(i), font_size, FONTS_PATH + unicode_mapping_dict[i][a]))
        except (ValueError, OSError) as e:
            print(e, i, len(unicode_mapping_dict[i]))


def display_chars(display_train, display_test, font_size=.2, img_size=400):
    anchors, positives, negatives, x1_test, x2_test, y_test = compile_datasets(display_train, display_test, font_size,
                                                                               img_size)
    for i in range(display_train):
        cv.imshow('anchor', anchors[i])
        cv.imshow('positive', positives[i])
        cv.imshow('negative', negatives[i])
        cv.waitKey(0)
        cv.destroyAllWindows()
    for i in range(display_test):
        cv.imshow('x1', x1_test[i])
        cv.imshow('x2', x2_test[i])
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    # Tests drawing each unicode character with a random font
    # test_drawing(.4,200)

    # With OpenCV, display 10 training triplets and 5 testing pairs
    display_chars(10, 5, .4, 200)
