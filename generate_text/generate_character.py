import cv2
from random import randint
import numpy as np
from PIL import ImageFont, ImageDraw, Image


def randomize_location(font_obj, chars, x_range, y_range, out_of_bounds_threshold=0):
    width = font_obj.getsize(chars)[0]
    height = font_obj.getsize(chars)[1]
    x_max = (x_range + out_of_bounds_threshold) - width
    y_max = (y_range + out_of_bounds_threshold) - height
    x_start = 0 - out_of_bounds_threshold
    y_start = 0 - out_of_bounds_threshold
    rand_x_coord = randint(x_start, x_max)
    rand_y_coord = randint(y_start, y_max)
    return rand_x_coord, rand_y_coord


def drawChar(img_size, chars, font_size, font_path, color=0):
    font = ImageFont.truetype(font_path, int(img_size * font_size))
    img_PIL = Image.fromarray(np.full((img_size, img_size), 255, dtype=np.uint8), mode='L')
    draw_PIL = ImageDraw.Draw(img_PIL)
    coordinates = randomize_location(font, chars, x_range=img_size, y_range=img_size)  # Top-left of character
    draw_PIL.text(coordinates, chars, font=font, fill=color)
    return np.array(img_PIL)


# Testing first on just showing a single tensor, then will work on iteration
def transformImg(img):
    rows, cols = img.shape
    assert rows == cols
    size = rows

    # Creates 3 random start and end points for the transformation
    randStart = np.float32([[0, 0], [0, size], [size, size]])
    randEnd = randStart + np.random.uniform(-0.1 * size, +0.1 * size, size=randStart.shape).astype(np.float32)

    # Applies the transformation to the given image
    matrix = cv2.getAffineTransform(randStart, randEnd)
    result = cv2.warpAffine(img, matrix, (cols, rows), borderValue=255)
    return result


# Assumes the tensor is an n*28*28*1 array of integers between 0 and 255 inclusive
# The function returns the array with each 28*28*1 tensor having an affine transformation performed on it
# Applies the random affine transformations to an entire tensor of images
def transformTensor(tensor):
    # Gets the number of images in the tensor
    n = np.shape(tensor)[0]
    for i in range(0, n):
        tensor[i] = transformImg(tensor[i])
    return tensor


def main():
    fontPath = "../fonts/ARIALUNI.ttf"
    IMG_SIZE = 28
    img = drawChar(IMG_SIZE, u"\u279D", 0.9, fontPath)
    img = transformImg(img)

    # img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)

    assert img.shape == (IMG_SIZE, IMG_SIZE)
    assert img.dtype == np.uint8

    cv2.imshow('Drawn character', img)
    # Display until space bar pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()