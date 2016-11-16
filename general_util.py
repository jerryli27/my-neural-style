import scipy.misc
import numpy as np
import math
import os, sys
from PIL import Image


def imread(path, shape = None):
    """
    :param path: path to the image
    :param shape: (Height, width)
    :return: np array with shape (height, width, 3)
    """
    if shape is None:
        return np.asarray(Image.open(path).convert('RGB'), np.float32)
    else:
        return np.asarray(Image.open(path).convert('RGB').resize((shape[1], shape[0])), np.float32)
    # return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)

def read_and_resize_images(dirs, height, width):
    image_1 = imread(dirs[0])
    # If there is no width and height, we automatically take the first image's width and height and apply to all the
    # other ones.
    if width is not None:
        if height is not None:
            target_shape = (height, width)
        else:
            target_shape = (int(math.floor(float(image_1.shape[0]) /
                                           image_1.shape[1] * width)), width)
    else:
        if height is not None:
            target_shape = (height, int(math.floor(float(image_1.shape[1]) /
                                                   image_1.shape[0] * height)))
        else:
            target_shape = (image_1.shape[0], image_1.shape[1])
    images = [imread(d, shape=target_shape) for d in dirs]


    return images



def read_and_resize_batch_images(dirs, height, width):
    images = [imread(dir, shape=(height, width)) for dir in dirs]
    return np.array(images)

def get_all_image_paths_in_dir(dir):
    assert(dir.endswith('/'))
    content_dirs = []
    for file_name in os.listdir(dir):
        base, ext = os.path.splitext(file_name)
        if ext == '.jpg' or ext == '.png':
            content_dirs.append(dir + file_name)
    return content_dirs

def get_global_step_from_save_dir(save_dir):
    return int(save_dir[save_dir.rfind("-")+1:])

def get_batch_from_np_list(np_list, start_index, batch_size):
    """

    :param np_list: a list of np arrays with size (1, height, width, depth)
    :param start_index:
    :param batch_size:
    :return: An np array with size = (batch, height, width, depth)
    """

    l = len(np_list)
    assert batch_size < l
    if start_index + batch_size < l:
        return np.concatenate(np_list[start_index:start_index+batch_size])
    else:
        end_index = (start_index + batch_size) % l
        return np.concatenate(np_list[start_index:] + np_list[:end_index])


def get_batch(dir_list, start_index, batch_size):
    """

    :param dir_list: a list of directories
    :param start_index:
    :param batch_size:
    :return: An array with length = batch.
    """

    l = len(dir_list)
    assert batch_size < l
    start_index = start_index % l
    if start_index + batch_size < l:
        return dir_list[start_index:start_index+batch_size]
    else:
        end_index = (start_index + batch_size) % l
        return dir_list[start_index:] + dir_list[:end_index]