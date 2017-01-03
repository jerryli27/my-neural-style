
"""
This file provides functions for turning any image into a 'sketch'.
"""

import random

import cv2
import numpy as np


def image_to_sketch(img):
    """
    :param image: An image represented in numpy array with shape (height, width, 3) or (batch, height, width, 3)
    :return: A sketch of the image with shape (height, width) or (batch, height, width)
    """

    if len(img.shape) == 4:
        img_diff_dilation_gray =  np.array([image_to_sketch(img[i,...]) for i in range(img.shape[0])])
        return img_diff_dilation_gray
    elif len(img.shape) == 3:

        kernel = np.ones((5, 5), np.uint8)
        img_dilation = cv2.dilate(img, kernel, iterations=1)

        img_diff_dilation = np.abs(np.subtract(img, img_dilation))
        img_diff_dilation_gray = cv2.cvtColor(img_diff_dilation, cv2.COLOR_RGB2GRAY)

        return img_diff_dilation_gray
    else:
        print('Image has to be either of shape (height, width, num_features) or (batch_size, height, width, num_features)')
        raise AssertionError


def generate_hint_from_image(img):
    """
    :param image: An image represented in numpy array with shape (height, width, 3) or (batch, height, width, 3)
    :return: A hint of the color usage of the original image image with shape (height, width, 4) or
    (batch, height, width, 4) where the last additional dimension stands for a (in rgba).
    """
    _max_num_hint = 20
    _hint_width = 5
    _hint_height = 5
    if len(img.shape) == 4:
        img_diff_dilation_gray =  np.array([generate_hint_from_image(img[i, ...]) for i in range(img.shape[0])])
        return img_diff_dilation_gray
    elif len(img.shape) == 3:
        num_hints = random.randint(0, _max_num_hint)
        height, width, rgb = img.shape
        assert rgb==3

        # All unmarked pixels are filled by 0,0,0,0 by default.
        ret = np.zeros(shape=(img.shape[0],img.shape[1],4))

        # Select random sites to give hints about the color used at that point.
        for hint_i in range(num_hints):
            rand_x = random.randint(0,width-1)
            rand_y = random.randint(0,height-1)
            ret[max(0, rand_y - _hint_height / 2):min(height, rand_y + _hint_height),max(0, rand_x - _hint_width / 2):min(width, rand_x + _hint_width),0:3] = img[max(0, rand_y - _hint_height / 2):min(height, rand_y + _hint_height),max(0, rand_x - _hint_width / 2):min(width, rand_x + _hint_width),:]
            ret[max(0, rand_y - _hint_height / 2):min(height, rand_y + _hint_height),max(0, rand_x - _hint_width / 2):min(width, rand_x + _hint_width),3] = np.ones((min(height, rand_y + _hint_height) - max(0, rand_y - _hint_height / 2),min(width, rand_x + _hint_width) - max(0, rand_x - _hint_width / 2))) * 255.0
        return ret

    else:
        print('Image has to be either of shape (height, width, num_features) or (batch_size, height, width, num_features)')
        raise AssertionError

