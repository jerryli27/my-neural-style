
"""
This file provides functions for turning any image into a 'sketch'.
"""

import random

import cv2
import numpy as np
from scipy.stats import threshold


def image_to_sketch(img):
    """
    :param image: An image represented in numpy array with shape (height, width, 3) or (batch, height, width, 3)
    :return: A sketch of the image with shape (height, width) or (batch, height, width)
    """

    # We must apply a lower threshold. Otherwise the sketch image will be filled with non-zero values that may provide
    # hints to the cnn trained. (It is unlikely to occur in human provided sketches that we have many pixels with
    # brightness lower than 32. )
    SKETCH_LOWEST_BRIGHTNESS = 32


    if len(img.shape) == 4:
        img_diff_dilation_gray =  np.array([image_to_sketch(img[i,...]) for i in range(img.shape[0])])
        return img_diff_dilation_gray
    elif len(img.shape) == 3:

        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(img, kernel, iterations=1)

        img_diff_dilation = np.abs(np.subtract(img, img_dilation))
        img_diff_dilation_gray = cv2.cvtColor(img_diff_dilation, cv2.COLOR_RGB2GRAY)



        img_diff_dilation_gray_thresholded = threshold(img_diff_dilation_gray, SKETCH_LOWEST_BRIGHTNESS)

        return img_diff_dilation_gray_thresholded
    else:
        print('Image has to be either of shape (height, width, num_features) or (batch_size, height, width, num_features)')
        raise AssertionError


def generate_hint_from_image(img):
    """
    :param image: An image represented in numpy array with shape (height, width, 3) or (batch, height, width, 3)
    :return: A hint of the color usage of the original image image with shape (height, width, 4) or
    (batch, height, width, 4) where the last additional dimension stands for a (in rgba).
    """
    _max_num_hint = 15
    _min_num_hint = 5
    _hint_max_width = 15
    _hint_min_width = 5
    _hint_max_area = 100
    _hint_min_area = 25

    if len(img.shape) == 4:
        img_diff_dilation_gray =  np.array([generate_hint_from_image(img[i, ...]) for i in range(img.shape[0])])
        return img_diff_dilation_gray
    elif len(img.shape) == 3:
        if _min_num_hint == _max_num_hint:
            num_hints = _max_num_hint
        else:
            num_hints = random.randint(_min_num_hint, _max_num_hint)


        height, width, rgb = img.shape
        assert rgb==3

        # All unmarked pixels are filled by 0,0,0,0 by default.
        ret = np.zeros(shape=(img.shape[0],img.shape[1],4))

        # Select random sites to give hints about the color used at that point.
        for hint_i in range(num_hints):
            curr_hint_width = random.randint(_hint_min_width, _hint_max_width)
            curr_hint_area = random.randint(_hint_min_area, _hint_max_area)
            curr_hint_height = int(curr_hint_area / curr_hint_width)

            rand_x = random.randint(0,width-1)
            rand_y = random.randint(0,height-1)
            ret[max(0, rand_y - curr_hint_height / 2):min(height, rand_y + curr_hint_height),max(0, rand_x - curr_hint_width / 2):min(width, rand_x + curr_hint_width),0:3] = img[max(0, rand_y - curr_hint_height / 2):min(height, rand_y + curr_hint_height),max(0, rand_x - curr_hint_width / 2):min(width, rand_x + curr_hint_width),:]
            ret[max(0, rand_y - curr_hint_height / 2):min(height, rand_y + curr_hint_height),max(0, rand_x - curr_hint_width / 2):min(width, rand_x + curr_hint_width),3] = np.ones((min(height, rand_y + curr_hint_height) - max(0, rand_y - curr_hint_height / 2),min(width, rand_x + curr_hint_width) - max(0, rand_x - curr_hint_width / 2))) * 255.0
        return ret

    else:
        print('Image has to be either of shape (height, width, num_features) or (batch_size, height, width, num_features)')
        raise AssertionError

