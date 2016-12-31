
"""
This file provides functions for turning any image into a 'sketch'.
"""

import cv2
import numpy as np

def image_to_sketch(img):
    """
    :param image: An image represented in numpy array with shape (height, width, 3)
    :return: A sketch of the image with shape (height, width)
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


