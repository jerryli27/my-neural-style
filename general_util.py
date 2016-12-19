import math
import os
from operator import mul

import numpy as np
import scipy.misc
from PIL import Image


def imread(path, shape = None, bw = False):
    """
    :param path: path to the image
    :param shape: (Height, width)
    :return: np array with shape (height, width, 3)
    """
    if shape is None:
        return np.asarray(Image.open(path).convert('L' if bw else 'RGB'), np.float32)
    else:
        return np.asarray(Image.open(path).convert('L' if bw else 'RGB').resize((shape[1], shape[0])), np.float32)
    # return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)

def read_and_resize_images(dirs, height, width):
    if isinstance(dirs, list):
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
    elif isinstance(dirs, str):
        image_1 = imread(dirs)
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
        return imread(dirs, shape=target_shape)



def read_and_resize_batch_images(dirs, height, width):
    images = [imread(dir, shape=(height, width)) for dir in dirs]
    return np.array(images)


def read_and_resize_bw_mask_images(dirs, height, width, batch_size, semantic_masks_num_layers):
    # Assume now that there is no order difference.
    images = [imread(dir, shape=(height, width), bw=True) for dir in dirs]
    np_images = np.array(images)
    np_images = np.reshape(np_images, (batch_size, semantic_masks_num_layers, height, width))
    np_images = np.transpose(np_images, (0, 2, 3, 1))
    return np_images

def get_all_image_paths_in_dir(dir):
    assert(dir.endswith('/'))
    content_dirs = []
    for file_name in os.listdir(dir):
        base, ext = os.path.splitext(file_name)
        if ext == '.jpg' or ext == '.png':
            content_dirs.append(dir + file_name)
    if (len(content_dirs) == 0):
        print('There is no image in directory %s' % dir)
        raise AssertionError
    content_dirs = sorted(content_dirs)
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
    if not batch_size <= l:
        print ('Given batch size must be smaller than the number of photos to load. Batch size : %d, num photos: %d'
               %(batch_size, len(dir_list)))
        raise AssertionError
    start_index = start_index % l
    if start_index + batch_size < l:
        return dir_list[start_index:start_index+batch_size]
    else:
        end_index = (start_index + batch_size) % l
        return dir_list[start_index:] + dir_list[:end_index]

def get_batch_indices(dir_list, start_index, batch_size):
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
        return range(start_index,start_index + batch_size)
    else:
        end_index = (start_index + batch_size) % l
        return range(start_index,l) + range(0,end_index)


def get_np_array_num_elements(arr):
    return reduce(mul, arr.shape, 1)

def np_image_dot_mask(image, mask):
    image_layer_num = image.shape[3]
    mask_layer_num = mask.shape[3]

    ret = []
    for i in range(image_layer_num):
        for j in range(mask_layer_num):
            ret.append(np.multiply(image[...,i], mask[...,j]))

    ret = np.transpose(np.array(ret), axes=(1,2,3,0))
    return ret

def resize_image_like_layers(image, shape):
    """

    :param image: An np array with shape batch, height, width, num_features
    :param shape: a tuple with format (new_height, new_width)
    :return: An np array with shape batch, new_height, new_width, num_features
    """
    resized_image = []
    for batch in range(image.shape[0]):
        resized_image.append([])
        for num_features in range(image.shape[3]):
            resized_image[-1].append(scipy.misc.imresize(image[batch,:,:,num_features], shape))

    resized_image = np.transpose(np.array(resized_image), (0,2,3,1))
    return resized_image

