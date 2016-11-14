import scipy
import numpy as np
import math

def imread(path):
    return scipy.misc.imread(path).astype(np.float)


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)

def read_and_resize_images(contents, styles, height, width):
    content_images = [imread(content) for content in contents]
    style_images = [imread(style) for style in styles]

    # If there is no width and height, we automatically take the first image's width and height and apply to all the
    # other ones.
    if width is not None:
        if height is not None:
            target_shape = (height, width)
        else:
            target_shape = (int(math.floor(float(content_images[0].shape[0]) /
                                           content_images[0].shape[1] * width)), width)
    else:
        if height is not None:
            target_shape = (height, int(math.floor(float(content_images[0].shape[1]) /
                                                   content_images[0].shape[0] * height)))
        else:
            target_shape = (content_images[0].shape[0], content_images[0].shape[1])

    for style_i in range(len(content_images)):
        if content_images[style_i].shape != target_shape:
            content_images[style_i] = scipy.misc.imresize(content_images[style_i], target_shape)
    for style_i in range(len(style_images)):
        style_images[style_i] = scipy.misc.imresize(style_images[style_i], (
            target_shape[0], target_shape[1]))
    return content_images, style_images

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