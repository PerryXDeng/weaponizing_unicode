import cv2
from random import randint
import numpy as np
from PIL import ImageFont, ImageDraw, Image


def build_28x28_white_image():
    img = np.zeros((28, 28, 3), np.uint8)
    img[:] = 255
    return img


def randomize_location(font_obj, chars, out_of_bounds_threshold=0, x_range=28, y_range=28):
    width = font_obj.getsize(chars)[0]
    height = font_obj.getsize(chars)[1]
    x_max = (x_range + out_of_bounds_threshold) - width
    y_max = (y_range + out_of_bounds_threshold) - height
    x_start = 0 - out_of_bounds_threshold
    y_start = 0 - out_of_bounds_threshold
    rand_x_coord = randint(x_start, x_max)
    rand_y_coord = randint(y_start, y_max)
    return rand_x_coord, rand_y_coord



def drawChar(chars, font_size, font_path, color=(0, 0, 0), base_image=build_28x28_white_image()):
    font = ImageFont.truetype(font_path, font_size)
    random_location_x, random_location_y = randomize_location(font, chars)
    img_PIL = Image.fromarray(base_image)
    draw_PIL = ImageDraw.Draw(img_PIL)
    coordinates = (random_location_x,random_location_y) # Top-left of character
    draw_PIL.text(coordinates, chars, font=font, fill=color)
    base_image = np.array(img_PIL)
    return base_image



# Testing first on just showing a single tensor, then will work on iteration
def transformImg(img):
    rows, cols, ch = img.shape

    # Creates 3 random start and end points for the transformation
    randStart = np.float32([[0, 0], [0, 27], [27, 27]])

    randEnd = np.float32([[randint(-3, 3), randint(-3, 3)], [randint(-3, 3), 27 - randint(-3, 3)], [27 - randint(-3, 3), 27 - randint(-3, 3)]])

    # Applies the transformation to the given image
    matrix = cv2.getAffineTransform(randStart, randEnd)
    result = cv2.warpAffine(img, matrix, (cols, rows), borderValue=(255, 255, 255))
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

# turns ..., 28, 28, 3 to ..., 28, 28, 1
def rgb2gray(rgb):
    return np.reshape(np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721]), (list(rgb.shape[:-1]) + [1]))


def main():
    fontPath = "../fonts/ARIALUNI.ttf"
    img = drawChar(u"\u279D", 12, fontPath)
    Image.fromarray(img, mode='RGB').show()
    img = rgb2gray(img)

    img = np.reshape(img, (28, 28))
    img.astype(np.int8, copy=False)
    img = np.reshape(img, (28, 28))
    #Image.fromarray(img, mode='L').show()
    return


if __name__ == "__main__":
    main()

