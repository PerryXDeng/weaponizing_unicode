import cv2
from random import randint
import numpy as np
from PIL import ImageFont, ImageDraw, Image, ImageFilter
from itertools import chain
import sys

from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode


def main():
    fontPath = "../fonts/ARIALUNI.ttf"

    img = drawChar(u"\u01c4", 12, fontPath)
    cv2.namedWindow("Character Display")  # Create a window for display.
    cv2.imshow("Character Display", img)  # Show our image inside it.
    cv2.waitKey(0)
    cv2.imshow("Affine Transformation display", transformImg(img))
    cv2.waitKey(0)                      # Wait for a keystroke in the window
    #
    # ttf = TTFont(sys.argv[1], 0, verbose=0, allowVID=0,
    #              ignoreDecompileErrors=True,
    #              fontNumber=-1)
    #
    # chars = chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables)
    # print(list(chars))
    #
    # # Use this for just checking if the font contains the codepoint given as
    # # second argument:
    # #char = int(sys.argv[2], 0)
    # #print(Unicode[char])
    # #print(char in (x[0] for x in chars))
    #
    # ttf.close()

    return 0



def build_28x28_white_image():
    img = np.zeros((28, 28, 3), np.uint8)
    img[:] = tuple(reversed((255, 255, 255)))
    return img


#
def randomize_location(font_obj, chars, out_of_bounds_threshold=0, x_range=28, y_range=28):
    width = font_obj.getsize(chars)[0]
    height = font_obj.getsize(chars)[1]
    print("Text width: " + str(width) + "\nText height: " + str(height))
    x_max = (x_range + out_of_bounds_threshold) - width
    y_max = (y_range + out_of_bounds_threshold) - height
    x_start = 0 - out_of_bounds_threshold
    y_start = 0 - out_of_bounds_threshold

    rand_x_coord = randint(x_start, x_max)
    rand_y_coord = randint(y_start, y_max)

    return rand_x_coord, rand_y_coord



def drawChar(chars, font_size, font_path, openCV=False, color=(0, 0, 0), base_image=build_28x28_white_image()):
    font = ImageFont.truetype(font_path, font_size)

    random_location_x, random_locaiton_y = randomize_location(font, chars)


    print("Random location of x: " + str(random_location_x) + "\nRandom location of Y: " + str(random_locaiton_y))
    img_PIL = Image.fromarray(base_image)
    draw_PIL = ImageDraw.Draw(img_PIL)
    coordinates = (random_location_x,random_locaiton_y) # Top-left of character
    draw_PIL.text(coordinates, chars, font=font, fill=color)
    base_image = np.array(img_PIL)

    return base_image


# Assumes the tensor is an n*28*28*1 array of integers between 0 and 255 inclusive
# The function returns the array with each 28*28*1 tensor having an affine transformation performed on it
# Testing first on just showing a single tensor, then will work on iteration
def transformImg(img):
    rows, cols, ch = img.shape

    # Creates 3 random start and end points for the transformation
    randStart = np.float32([[0, 0], [0, 27], [27, 27]])

    randEnd = np.float32([[randint(0, 10), randint(0, 10)], [randint(0, 10), 27 - randint(0, 10)], [27 - randint(0, 10), 27 - randint(0, 10)]])

    # Applies the transformation to the given image
    matrix = cv2.getAffineTransform(randStart, randEnd)
    result = cv2.warpAffine(img, matrix, (cols, rows), borderValue=(255, 255, 255))
    return result

main()

